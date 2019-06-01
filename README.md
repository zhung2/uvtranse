# Union Visual Translation Embedding

## Dataset
* Download the dataset for [VRD](http://cs.stanford.edu/people/ranjaykrishna/vrd/dataset.zip) and put it in `$ROOT/data` directory
* Download and Extract features that are obtained by detectors and put them also in `$ROOT/data`:
  * [train](https://uofi.box.com/s/elrnih13mimgt9t7ey2bsp95dqa74ut1)
  * [val](https://uofi.box.com/s/e926m79ure6mku3400rh6bfw9ptujhfa)
  * [test_part1](https://uofi.box.com/s/xx4j8x9u309c6uz3s4qxe2ou5dyk4b3m)
  * [test_part2](https://uofi.box.com/s/yfvtfgloc71qzwm1rvfk7m298q92t2wj)
  * [test_part3](https://uofi.box.com/s/fxpsuw2ac3yq85ayvqlob3iqekbhkqcx)
  * [test_part4](https://uofi.box.com/s/awwpiks68qyvq932svlvtj7o9sxhnoz6)
  * [Object Glove Embedding](https://uofi.box.com/s/42h03v8v1zgul7ae8qyuubp6dovrt3vq)

* Overall structure
  |---- vrd<br/>
  |-------- annotation_train.mat<br/>
  |-------- annotation_test.mat<br/>
  |-------- objectListN.mat<br/>
  |-------- predicate.mat<br/>
  |-------- images<br/>
  |------------ sg_train_images<br/>
  |------------ sg_test_images<br/>
  |-------- test<br/>
  |------------ multi_vgg16_test_dict.pkl<br/>
  |-------- train<br/>
  |------------ multi_vgg16_train_dict.pkl<br/>
  |-------- val<br/>
  |------------ multi_vgg16_val_dict.pkl<br/>

## Usage
* Training
```python
python train.py 
```
Also check the available arguments inside `options/`

* Testing

First generate the matlab format detection
```python
python test.py
```
Also check the available arguments inside `options/`

Then use the (official evaluation code)[https://github.com/Prof-Lu-Cewu/Visual-Relationship-Detection]


## Model
* [Visual only](https://uofi.box.com/s/vm5a419n3u97jlygva7uxaf8wkxaqj6c)
* [Visual + Language](https://uofi.box.com/s/iajcwh2u2pfbfv1g0yagwjst17pax9l5)
