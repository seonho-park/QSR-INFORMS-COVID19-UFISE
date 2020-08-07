    # from lungmask import mask
    # import SimpleITK as sitk
    # from PIL import Image
    # INPUT = "/home/sean/data/COVID-CT QSR Data Challenge/COVID-CT QSR Data Challenge/Images-processed/CT_COVID/2020.02.10.20021584-p6-52%4.png"
    # input_image = sitk.ReadImage(INPUT)
    # segmentation = mask.apply(input_image)
    import nibabel as nib
    img = nib.load("/home/sean/data/COVID-CT QSR Data Challenge/LungSegCT/tr_im.nii.gz")
    
