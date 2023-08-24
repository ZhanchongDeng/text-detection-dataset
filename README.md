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
- [x] [COCO-Text](https://vision.cornell.edu/se3/coco-text-2/)
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
- [x] [TextOCR](https://textvqa.org/textocr/dataset/)
    * 24902 images, 1202339 text instances
    * does not differentiate between illegible/non-english
    * inconsistent partial word crop (word??? can be detect with box only including word??, missing the last ?)
- [x] [UberText](https://s3-us-west-2.amazonaws.com/uber-common-public/ubertext/index.html)
    * 117969 images, 571534 text instances
    * inconsistently bound space separated as one box
    * contains partial illegible text
- [ ] [SynthText](https://www.robots.ox.ac.uk/~vgg/data/scenetext/)
- [ ] [ICDAR 2017 MLT](https://rrc.cvc.uab.es/?ch=8&com=introduction)
    * mostly non English
    * no differentiation

\* TotalText and CTW-1500 are included in ArT dataset.

## Result
| dataset_name   |   num_images |   num_text_instances |   num_illegible |   num_partial_illegible |   percent_legible |   avg_text_per_image |
|:---------------|-------------:|---------------------:|----------------:|------------------------:|------------------:|---------------------:|
| SVT            |          349 |        904           |               0 |                       0 |          1        |              2.59026 |
| ICDAR2013      |          462 |       1944           |               0 |                       0 |          1        |              4.20779 |
| MSRA-TD500     |          500 |       1719           |               0 |                       0 |          1        |              3.438   |
| ICDAR2015      |         1500 |      17116           |           10571 |                       0 |          0.382391 |             11.4107  |
| ArT            |         5049 |      49179           |           12900 |                       0 |          0.737693 |              9.74034 |
| COCO_Text      |        19938 |      94711           |           10778 |                       0 |          0.886201 |              4.75028 |
| TextOCR        |        24902 |          1.20234e+06 |          379767 |                       0 |          0.684143 |             48.2828  |
| UberText       |       117969 |     571534           |          252181 |                   48401 |          0.474079 |              4.84478 |
| Total          |       170669 |          1.93945e+06 |          666197 |                   48401 |          0.631545 |             11.3638  |

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