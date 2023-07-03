---
layout: homepage
---

# Summary
<p align="justify">
Our primary focus is on developing well-calibrated out-of-distribution (OOD) detectors to ensure the safe deployment of medical image classifiers. The use of synthetic augmentations has become common for specifying regimes of data inliers and outliers. However, our research findings highlight the substantial influence of both the synthesis space and the type of augmentation on the performance of OOD detectors. After conducting an extensive study using medical imaging benchmarks and open-set recognition settings, we recommend employing a combination of virtual inliers in the classifier's latent space and diverse synthetic outliers in the pixel space. This approach proves highly effective in producing OOD detectors with superior performance.
</p>

# Video
{% include add_video.html 
    youtube_link="https://www.youtube.com/embed/jpR7ouFTDqA" 
%}


# Method

{% include add_image.html 
    image="assets/img/approach.png"
    caption="" 
    alt_text="Alt text" 
%}


<!-- <div style="font-size:18px">
  <ol type="a">
  <p align="justify">
  <li><strong>Training:</strong> Train the medical image classifier along with the appropriate calibration protocol i.e latent-space inliers & pixel-space outliers.</li>
  <li><strong>OOD Detection:</strong> Use an energy based OOD detector to distinguish between ID and OOD (Modality Shifts / Novel Classes) and compute the performance metrics for e.g., AUROC</li>
  </p>
</ol>
</div> -->


<!-- {% include add_image.html 
    image="assets/img/website-fig-teaser.png"
    caption="Examples of synthetic data generated using SiSTA. <strong>Please follow the link by clicking the image</strong> to access additional examples for different benchmarks and distribution shifts." 
    alt_text="Alt text" 
    link="https://icml-sista.github.io/"
    height="400"
%} -->



# Empirical Results


{% include add_gallery.html data="results" %}



# Citation

{% include add_citation.html text="@inproceedings{narayanaswamy2023know,
title={Know Your Space: Inlier and Outlier Construction for Calibrating Medical {OOD} Detectors},
author={Vivek Narayanaswamy and Yamen Mubarka and Rushil Anirudh and Deepta Rajan and Andreas Spanias and Jayaraman J. Thiagarajan},
booktitle={Medical Imaging with Deep Learning},
year={2023}}" %}


# Contact
If you have any questions, please feel free to contact us via email: {{ site.contact.emails }}
