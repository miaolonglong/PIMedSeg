# 2023CMPB：PIMedSeg:渐进式交互式医学图像分割 
Gong X, Wang L, Miao L, et al. PIMedSeg: Progressive Interactive Medical Image Segmentation[J]. Computer Methods and Programs in Biomedicine, 2023: 107776.
[论文地址](https://www.sciencedirect.com/science/article/abs/pii/S016926072300442X)

医学图像中准确的对象分割是医学诊断和其他应用中的关键步骤。尽管对自动分割方法进行了多年的研究，但实现临床可接受的图像质量仍然具有挑战性。交互式分割被视为一种有前途的替代方案；因此，我们提出了一种基于渐进式工作流程的新的交互式分割框架，以减少用户的工作量并提供高质量的结果。首先，我们的方法使用我们提出的磁盘和曲线变换对用户提供的区域点击和边缘涂鸦进行编码。然后，使用基于变压器的模块进行细化，该模块从卷积神经网络（CNN）的输出和额外的输入映射中提取有效特征。通过对各种医学图像进行了大量的实验，包括超声波（US）、计算机断层扫描（CT）和磁共振图像（MRI），已经证明了我们的新方法相对于最先进的替代方法的有效性。我们所提出的框架可以使用最少的交互实现高质量的分割，而无需花费大量的手动分割成本。
![figure2](/figure/图片2.png)
# Introduction
We propose a medical image segmentation framework called PIMedSeg, which combines region click and edge scribble interactions. Region clicks are user-friendly but can become limiting with more rounds of interaction. To address this, we introduced a second pattern where users provide both edge scribbles and region clicks. We represent this interactive information using a novel disk and curve (DAC) transform. Additionally, we developed TFineNet, a transformer-based module, to extract high-level features with exact semantics and capture long-range dependencies from the interactive information.
![figure1](/figure/图片1.png)

*	We propose a progressive workflow to enhance segmentation quality and minimize user effort. We utilize region clicks for rapid segmentation, and employ edge scribbles along with region clicks to update areas that are difficult to segment.
*	We introduce TFineNet, a powerful yet straightforward transformer-based module that generates highly effective feature representations with precise semantics.
*	The proposed model is lightweight and generalizable, aligning with the characteristics of desirable interactive segmentation. Extensive experiments have been conducted on a variety of medical images, including ultrasound (US), computerized tomography (CT) and magnetic resonance images (MRI), to verify the effectiveness of our new approach over state-of-the-art methods.

# Result
We compared PIMedSeg with GraphCut, Random Walks, Slic-Seg, DeepIGeoS, and MIDeepSeg for spleen segmentation, and the results showed that PIMedSeg achieves higher accuracy with a lower user time than the other methods.
| Method | Dice(%) | ASSD(mm) | HD(mm) | Time(s) | Points(pixels) |
|-----:|-----------|-----------|-----------|-----------|-----------|
|     GraphCut| 95.27±4.36 | 1.30±0.42 |  -  | 21.2±7.7 | 335.1±91.7 |
|     Random Walks| 95.51±1.59   | 1.45±2.66 |- |20.1±7.9 | 218.4±69.0 |
|     Slic-Seg|      95.18±4.70   |1.23±0.39 |- |20.1±8.2| 254±77.5| 
|     DeepIGeoS |    96.39±2.22   | 1.71±2.74     | -     |6.1±4.8      | 31.1±52.4   |
|     MIDeepSeg|     96.93±1.43   | 1.18±0.44   | -| 4.76±2.0|**4.85±1.6**|
|     Ours w/o ES|97.15±2.08 | 0.67±0.12| 1.99±0.39| **1.7±0.1*** | 20±0 |
|     Ours with ES| **98.33±1.27*** | **0.41±0.12***|  1.71±0.48 | 2.1±0.1| 198.8±85.9|

* Visualization of interactive segmentation for the spleen CT from the BTCV+TCIA dataset. The red box in the first column is used to indicate the interior area to be zoomed in for greater clarity. The second and third columns show the results obtained using region clicks, and the last two columns show the results obtained using region clicks and edge scribbles. 5th and 10th mean the fifth and tenth rounds of user interactions, respectively.

![figure5](/figure/图片5.jpg)

* Visualization of interactive segmentation for the breast lesion US (1st row), liver CT (2nd row), and brain tumor MRI (3rd row). The red box in the first subfigure is used to indicate the interior area to be zoomed in for greater clarity. The second and third columns show the results obtained using region clicks, and the last two columns show the results obtained using region clicks and edge scribbles. 5th and 10th mean the fifth and tenth rounds of user interactions, respectively.

![figure7](/figure/图片7.jpg)
# Conclusion
* **Framework Introduction**: Presented a progressive interactive segmentation framework for medical images.
   
* **User Input Encoding**: Utilized a novel transformation method to encode user input, providing guidance to the segmentation network.

* **Semantic Context Enhancement**: Integrated a refinement module based on transformers to enhance the model's ability to extract semantic contexts, leading to high-quality segmentation with minimal user interactions.

# Future Directions
Discussed plans for future studies to enhance the model's perceptual ability with actual user inputs and explore the development of a unified framework for both 2D and 3D processing.
