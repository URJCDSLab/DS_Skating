# Figure Skating Pose Detection and 3D reconstruction

## Overview

This repository contains the implementation of a novel framework for automatic pose detection and projection to a 3D space based on  [METRABS](https://arxiv.org/abs/2007.07227)

<a href="url"><img src="https://github.com/user-attachments/assets/5d01ad28-99c5-4d7c-8651-e95ca4dddc2a" align="center" height="312" width="600" ></a>




## Authors

- Álvaro Novillo (0009-0003-9888-6638)
- Víctor Aceña (0000-0003-1838-2150)
  
Rey Juan Carlos University, Data Science Laboratory, C/ Tulipán, s/n, 28933, Móstoles, Spain
{alvaro.novillo, victor.acena}@urjc.es


The framework consists of a single module capable of detecting the skater, and extract its pose estimation in each frame. We could improve the player detection using another model such as Yolov8, and feeding those detections to METRABS



## Installation

1. Clone this repository:
   ```sh
   git clone https://github.com/AlvaroNovillo/DS_Skating.git
