import napari
import numpy as np


def mask_with_napari(image_3d: np.ndarray,
                    mask_3d: np.ndarray,
                    label: int = 1
                    ) -> np.ndarray:
    """
    Launches Napari to manually edit a 3D mask over a 3D image.

    Parameters:
    - image_3d: numpy.ndarray
        A 3D image array (shape: Z, Y, X)
    - mask_3d: numpy.ndarray
        A 3D mask array (shape: Z, Y, X)
    - label: label_map integer (e.g., 1 for vertebrae in map = {1: 'vertebrae'})

    Returns:
    - edited_mask: numpy.ndarray
        The modified mask after editing in Napari.
    """
    if image_3d.shape != mask_3d.shape:
        raise ValueError("Image and mask must have the same shape.")

    # Ensure label image is integer type
    mask_3d = mask_3d.astype(np.int32)
    
    # Launch Napari
    viewer = napari.Viewer()

    # Display the image and the mask
    viewer.add_image(image_3d, name='Image') # anatomical base layer
    mask_layer = viewer.add_labels(mask_3d, name='Mask') # shared label array

    # Set default painting label
    mask_layer.selected_label = label
    mask_layer.show_selected_label = True

    mask_layer.mode = 'paint'
    mask_layer.opacity = 0.4
    mask_layer.brush_size = 4

    # Set 2D slicing and coronal orientation for (X, Y, Z) image
    viewer.dims.ndisplay = 2
    viewer.dims.order = [1, 2, 0]  # Y, Z, X order for axial view

    # Run Napari event loop
    napari.run()

    # Return the edited mask
    return mask_layer.data
