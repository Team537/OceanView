# FRC-OAKD-Integration
This repository contains team 537's vision code for the 2024-2025 "REAFSCAPE" season. This software is intended to be run on a coprocessor such as a Raspberry PI, Orange PI, or Jetson Nano. To any other teams looking at this repository, welcome! If you wish to use this repository, all we ask is that you properly credit us.

## Setup / Installation
[To be added]

## Competition Sensor Calibration Procedure
At the start of each competition, the color threshold ranges will have to be retuned. Currently, this can be done by plugging the camera into a computer, running this code on the host device, and snapping a few images of the game elements that you wish to detect. From there, you can use sites like [pseudopencv](https://pseudopencv.site/utilities/hsvcolormask/) to help you tune the HSV threshold values.

From there, all that's required is changing the upper and lower bound variables in the code and changing them to the tuned values.
