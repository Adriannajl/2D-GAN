## Sample Preprocessing Steps

1. **Image Screening**  
   Due to the short sampling period between frames, many images in the original dataset were highly similar. To prevent model overfitting, a subset of representative samples was selected.

2. **Multi-view Composition**  
   Each selected sample included infrared images from four perspectives:
   - Face  
   - Eyes  
   - Hands  
   - Side view

3. **Data Augmentation**  
   The following techniques were applied to enhance data diversity:
   - Horizontal flipping  
   - Rotation  
   - Cropping  
   - Partial occlusion  
   - Blurring

4. **Example Visualization**  
   An example of augmented facial infrared images is shown in **Figure 6**.

5. **Final Dataset Composition**  
   - Total samples: **1120**  
   - Drinking samples: **640**  
   - Non-drinking samples: **480**

6. **Dataset Split**  
   The dataset was divided into:
   - **Training set**: 70%  
   - **Validation set**: 20%  
   - **Test set**: 10%
