def window_image(img, window_center=30, window_width=80, intercept=-1024.0, slope=1.0):
    img = img * slope + intercept
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    slice_s=img
#    print(slice_s.shape)
    slice_s = (slice_s - img_min)*(255/(img_max-img_min)) #or slice_s = (slice_s - (w_level-(w_width/2)))*(255/(w_width))
    slice_s[slice_s < 0]=0
    slice_s[slice_s > 255] = 255

    return slice_s
def window_ct (ct_scan, w_level=35, w_width=80):
    w_min = w_level - w_width / 2
    w_max = w_level + w_width / 2
    slice_s=ct_scan
    slice_s = (slice_s - w_min)*(255/(w_max-w_min)) #or slice_s = (slice_s - (w_level-(w_width/2)))*(255/(w_width))
    slice_s[slice_s < 0]=0
    slice_s[slice_s > 255] = 255
    return slice_s
