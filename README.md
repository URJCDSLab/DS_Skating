# Padel 2D-Tracking from Monocular Video

## Overview

This repository contains the implementation of a novel framework for automatic pose detection and projection to a 3D space based on  [METRABS](https://arxiv.org/abs/2007.07227)

<a href="url"><img src="https://github.com/AlvaroNovillo/DS_Padel/assets/81865790/a8435924-7e32-45fb-b4b7-e6eef7d03843" align="center" height="720" width="1280" ></a>



## Authors

- ÁLvaro Novillo (0009-0003-9888-6638)
Rey Juan Carlos University, Data Science Laboratory, C/ Tulipán, s/n, 28933, Móstoles, Spain
{alvaro.novillo}@urjc.es


The framework consists of a single module capable of detecting the skater, and extract its pose estimation in each frame. We could improve the player detection using another model such as Yolov8, and feeding those detections to METRABS



## Installation

1. Clone this repository:
   ```sh
   git clone https://github.com/AlvaroNovillo/DS_Skating.git
