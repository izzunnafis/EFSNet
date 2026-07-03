def correlation_volume(left_feature, right_feature, max_disp):
    b, c, h, w = left_feature.size()
    cost_volume = left_feature.new_zeros(b, max_disp, h, w)
    for i in range(max_disp):
        if i > 0:
            cost_volume[:, i, :, i:] = (left_feature[:, :, :, i:] * right_feature[:, :, :, :-i]).mean(dim=1)
        else:
            cost_volume[:, i, :, :] = (left_feature * right_feature).mean(dim=1)
    cost_volume = cost_volume.contiguous()
    return cost_volume


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume
