import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import Scale, normal_init

from numpy import pi

from mmdet.core import (MultiScale, distance2bbox, distance2obb,
                        force_fp32, mintheta_obb, multi_apply,
                        multiclass_arb_nms)

from mmdet.models.builder import HEADS, build_loss

from .obb_anchor_free_head import OBBAnchorFreeHead

INF = 1e8


@HEADS.register_module()
class FCOSFHead(OBBAnchorFreeHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 center_sampling=False,
                 center_sample_radius=1.5,
                 centerness_on_reg=False,
                 centerness_pow=0.2,
                 num_fourier_pairs=4,
                 norm_trig=False,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_trig=dict(type='L1Loss', loss_weight=0.1),
                 loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 **kwargs):

        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.centerness_on_reg = centerness_on_reg
        self.centerness_pow = centerness_pow
        self.num_fourier_pairs = num_fourier_pairs
        self.norm_trig = norm_trig
        self.trig_dim = 2

        super().__init__(
            num_classes,
            in_channels,
            bbox_type='obb',
            reg_dim=1+2*num_fourier_pairs,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            norm_cfg=norm_cfg,
            **kwargs)
        self.loss_trig = build_loss(loss_trig)
        self.loss_centerness = build_loss(loss_centerness)

        theta_offsets = torch.tensor([2., 1., 0., 3.]) * pi/2
        theta_multiplier = torch.arange(1, self.num_fourier_pairs+1).float()
        self.register_buffer('theta_offsets', theta_offsets.view(-1, 1))
        self.register_buffer('theta_multiplier', theta_multiplier.view(1, -1))

    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.conv_trig = nn.Conv2d(self.feat_channels, self.trig_dim, 3, padding=1)
        self.scales = nn.ModuleList([MultiScale(self.reg_dim, 1.0) for _ in self.strides])
        if not self.norm_trig:
            self.scale_t = Scale(1.0)

    def init_weights(self):
        """Initialize weights of the head."""
        super().init_weights()
        normal_init(self.conv_centerness, std=0.01)
        normal_init(self.conv_trig, std=0.01)

    def forward(self, feats):
        """Forward features of the head"""
        return multi_apply(self.forward_single, feats, self.scales,
                           self.strides)

    def forward_single(self, x, scale, _):
        """Forward features of a single scale level."""

        cls_score, reg_pred, cls_feat, reg_feat = super().forward_single(x)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        reg_pred = scale(reg_pred).float()
        # norm trig
        trig_pred = self.conv_trig(reg_feat)

        if self.norm_trig:
            trig_pred = trig_pred / torch.sqrt(
                trig_pred.square().sum(dim=1, keepdim=True))
        else:
            trig_pred = self.scale_t(trig_pred)

        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)
        return cls_score, reg_pred, trig_pred, centerness

    @force_fp32(apply_to=('cls_scores', 'reg_preds', 'trig_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             reg_preds,
             trig_preds,
             centernesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            reg_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            trig_preds (list[Tensor]): thetas delta for each scale
                level, each is a 1D-tensor, the channel number is
                num_points * 1.
            centernesses (list[Tensor]): Centerss for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        assert len(cls_scores) == len(reg_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, reg_preds[0].dtype,
                                           reg_preds[0].device)
        labels, bbox_targets = self.get_targets(all_level_points, gt_bboxes,
                                                gt_labels)
        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_reg_preds = [
            reg_pred.permute(0, 2, 3, 1).reshape(-1, self.reg_dim)
            for reg_pred in reg_preds
        ]
        flatten_trig_preds = [
            trig_pred.permute(0, 2, 3, 1).reshape(-1, self.trig_dim)
            for trig_pred in trig_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_reg_preds = torch.cat(flatten_reg_preds)
        flatten_trig_preds = torch.cat(flatten_trig_preds)
        flatten_centerness = torch.cat(flatten_centerness)

        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = len(pos_inds)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels,
            avg_factor=num_pos + num_imgs)  # avoid num_pos is 0

        pos_trig_preds = flatten_trig_preds[pos_inds]
        pos_reg_preds = flatten_reg_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]

        if num_pos > 0:
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_centerness_targets = self.centerness_target(pos_bbox_targets)
            loss_centerness = self.loss_centerness(pos_centerness,
                                                   pos_centerness_targets)
            pos_dist_targets, pos_angle_targets = torch.split(
                pos_bbox_targets, [4, 1], dim=1)

            # trigonometric loss
            sin_targets = torch.sin(4 * pos_angle_targets)
            cos_targets = torch.cos(4 * pos_angle_targets)
            pos_trig_targets = torch.cat([sin_targets, cos_targets], dim=1)

            loss_trig = self.loss_trig(pos_trig_preds, pos_trig_targets)
                # bbox loss using forier series
            pos_points = flatten_points[pos_inds]

            pos_reg_preds = self.fourier_decode(pos_reg_preds, pos_angle_targets)
            decoded_bbox_preds = distance2bbox(pos_points, pos_reg_preds)
            decoded_bbox_targets = distance2bbox(pos_points, pos_dist_targets)
            loss_bbox = self.loss_bbox(
                decoded_bbox_preds,
                decoded_bbox_targets,
                weight=pos_centerness_targets,
                avg_factor=pos_centerness_targets.sum())
        else:
            loss_trig = pos_trig_preds.sum()
            loss_centerness = pos_centerness.sum()
            loss_bbox = pos_reg_preds.sum()

        total_loss = dict(
            loss_cls = loss_cls,
            loss_trig = loss_trig,
            loss_centerness = loss_centerness,
            loss_bbox = loss_bbox
        )

        return total_loss 

    @force_fp32(apply_to=('cls_scores', 'reg_preds', 'trig_preds', 'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   reg_preds,
                   trig_preds,
                   centernesses,
                   img_metas,
                   cfg=None,
                   rescale=None):
        """ Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_points * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W)
            centernesses (list[Tensor]): Centerness for each scale level with
                shape (N, num_points * 1, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of the
                corresponding box.
        """
        assert len(cls_scores) == len(reg_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, reg_preds[0].dtype,
                                      reg_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            reg_pred_list = [
                reg_preds[i][img_id].detach() for i in range(num_levels)
            ]
            trig_pred_list = [
                trig_preds[i][img_id].detach() for i in range(num_levels)
            ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self._get_bboxes_single(cls_score_list,
                                                 reg_pred_list,
                                                 trig_pred_list,
                                                 centerness_pred_list,
                                                 mlvl_points, img_shape,
                                                 scale_factor, cfg, rescale)
            result_list.append(det_bboxes)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           reg_preds,
                           trig_preds,
                           centernesses,
                           mlvl_points,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                Has shape (num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (num_points * 4, H, W).
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (num_points * 4, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(reg_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        for cls_score, reg_pred, trig_pred, centerness, points, stride, in zip(
            cls_scores, reg_preds, trig_preds, centernesses, mlvl_points,
            self.strides):
            assert cls_score.size()[-2:] == reg_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            trig_pred = trig_pred.permute(1, 2, 0).reshape(-1, self.trig_dim)
            reg_pred = reg_pred.permute(1, 2, 0).reshape(-1, self.reg_dim)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                trig_pred = trig_pred[topk_inds, :]
                reg_pred = reg_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
            trig_pred = torch.atan2(trig_pred[:, [0]], trig_pred[:, [1]]) / 4
            reg_pred = self.fourier_decode(reg_pred, trig_pred) * stride

            bbox_pred = torch.cat([reg_pred, trig_pred], dim=1)
            bboxes = distance2obb(points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
        mlvl_bboxes = torch.cat(mlvl_bboxes)

        if rescale:
            scale_factor = mlvl_bboxes.new_tensor(scale_factor)
            mlvl_bboxes[..., :4] = mlvl_bboxes[..., :4] / scale_factor
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)

        det_bboxes, det_labels = multiclass_arb_nms(
            mlvl_bboxes,
            mlvl_scores,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img,
            score_factors=mlvl_centerness,
            bbox_type='obb',
            extra_dets=None)
        return det_bboxes, det_labels

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Get points according to feature map sizes."""
        y, x = super()._get_points_single(featmap_size, stride, dtype, device)
        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride),
                             dim=-1) + stride // 2
        return points

    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]
    
        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            bbox_targets[:, :4] = bbox_targets[:, :4] / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
        return concat_lvl_labels, concat_lvl_bbox_targets

    def _get_target_single(self, gt_bboxes, gt_labels, points, regress_ranges,
                           num_points_per_lvl):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)

        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.background_label), \
                   gt_bboxes.new_zeros((num_points, 5))

        areas = gt_bboxes[:, 2] * gt_bboxes[:, 3]
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        points = points[:, None, :].expand(num_points, num_gts, 2)
        gt_bboxes = mintheta_obb(gt_bboxes)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 5)
        gt_ctr, gt_wh, gt_thetas = torch.split(
            gt_bboxes, [2, 2, 1], dim=2)

        Cos, Sin = torch.cos(gt_thetas), torch.sin(gt_thetas)
        Matrix = torch.cat([Cos, -Sin, Sin, Cos], dim=-1).reshape(
            num_points, num_gts, 2, 2)
        offset = points - gt_ctr
        offset = torch.matmul(Matrix, offset[..., None])
        offset = offset.squeeze(-1)

        W, H = gt_wh[..., 0], gt_wh[..., 1]
        offset_x, offset_y = offset[..., 0], offset[..., 1]
        left = W / 2 + offset_x
        right = W / 2 - offset_x
        top = H / 2 + offset_y
        bottom = H / 2 - offset_y
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        # condition1: inside a gt bbox
        # if center_sampling is true, also in center bbox.
        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
        if self.center_sampling:
            # inside a `center bbox`
            radius = self.center_sample_radius
            stride = offset.new_zeros(offset.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            inside_center_bbox_mask = (abs(offset) < stride).all(dim=-1)
            inside_gt_bbox_mask = torch.logical_and(
                inside_center_bbox_mask, inside_gt_bbox_mask)

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.background_label  # set as BG
        bbox_targets = bbox_targets[range(num_points), min_area_inds]

        theta_targets = gt_thetas[range(num_points), min_area_inds]
        bbox_targets = torch.cat([bbox_targets, theta_targets], dim=1)
        return labels, bbox_targets

    def centerness_target(self, pos_bbox_targets):
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        centerness_targets = (
            left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.pow(centerness_targets, self.centerness_pow)

    def fourier_decode(self, fourier_coeff, theta_preds):
        theta = (theta_preds[..., None] + self.theta_offsets) * self.theta_multiplier
        Cos, Sin = torch.cos(theta), torch.sin(theta)
        One = Cos.new_ones((theta.size(0), 4, 1))
        trig_vector = torch.cat([One, Cos, Sin], dim=-1)
        decoded_dist = torch.matmul(trig_vector, fourier_coeff[..., None]).squeeze(-1) # [n, 4, 9] x [n, 9, 1]
        #  return torch.where(decoded_dist > 0, decoded_dist + 1, decoded_dist.exp())
        return F.relu(decoded_dist)
