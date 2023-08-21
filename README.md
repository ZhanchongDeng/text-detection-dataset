# Text Detection Dataset
Aggregate commonly used text detection datasets into a simplified COCO format.

Supported Dataset:
- [x] [ICDAR 2013](https://rrc.cvc.uab.es/?ch=2&com=introduction)
    * 462 images, 1944 text instances
    * mostly horizontal
    * sometimes ignore smaller text
- [x] [ICDAR 2015](https://rrc.cvc.uab.es/?ch=4&com=introduction)
    * 1500 images, 17116 text instances 
    * a lot of ### (meaning unrecognizable)
    * image often taken while moving
    * upper/lower case error present (1 out of 50)
    * inconsistently bound space separated as one box
- [x] [COCO-Text](https://rrc.cvc.uab.es/?ch=5&com=downloads)
    * 19938 images, 94711 text instances
    * half of the images does not have annotations
- [x] [MSRA-TD500](http://www.iapr-tc11.org/mediawiki/index.php/MSRA_Text_Detection_500_Database_(MSRA-TD500))
    * 500 images, 1719 text instances
    * only for detection, no labeled text
    * does not differentiate language
    * inconsistently bound space separated as one box
- [x] [SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset)
    * 349 images, 904 text instances
    * mostly horizontal
    * all CAPS
    * inconsistently ignore text on screen
- [x] [ICDAR 2019 ArT](https://rrc.cvc.uab.es/?ch=14&com=downloads)
    * 5049 images, 49179 text instances
    * 20% illegible
    * contains non English text
- [ ] [SynthText](https://www.robots.ox.ac.uk/~vgg/data/scenetext/)
- [ ] [ICDAR 2017 MLT](https://rrc.cvc.uab.es/?ch=8&com=introduction)
    * mostly non English
    * no differentiation

\* TotalText and CTW-1500 are included in ArT dataset.
## Build
First clone the repo,
```
git clone --recursive git@github.com:eyepop-ai/text-detection-dataset.git
```
and install the necessary packages
```
cd text-detection-dataset
make configure
```