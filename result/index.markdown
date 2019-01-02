---
layout: page
title: SADL Experiment Result
---

This page accompanies a submission to ICSE 2019 Technical Papers track, "Guiding Deep Learning System Testing Using Surise Adequacy". We have listed all figures, including the ones omitted from the submission due to space limit (<span style="color:red;">titles of figures that are not in the paper are in colour *red*</span>). The page also contains additional analysis undertaken as part of author response.

# RQ1: Is SADL capable of capturing the relative surprise of an input of a DL system?

#### Figure 2 

All results were included in the paper.

{% include image.html url="./figures/RQ1_test_accuracy.png" description="Accuracy of test inputs in MNIST and CIFAR-10 dataset, selected from the input with the lowest SA, increasingly including inputs with higher SA, and vice versa  (i.e., from the input with the highest SA to inputs with lower SA)." %}

#### Figure 4

We have included the DSA plots for MNIST and CIFAR-10 in the paper.

{% include image.html url="./figures/RQ1_sorted_dsa.png" description="Sorted DSA values of adversarial examples for MNIST and CIFAR-10." %}

#### <span style="color:red">Figure 4' (NOT IN THE PAPER)</span>

In addition, here are the per-class plots that show sorted DSA values of each class in MNIST. Note that the number of adversarial examples of each class is different because each adversarial example generation algorithm has own method of targeting specific class.

{% include image.html url="./figures/RQ1_sorted_dsa_mnist_class.png" description="Sorted DSA values of adversarial examples for MNIST-10 per class." %}

#### <span style="color:red">Figure 4'' (NOT IN THE PAPER)</span>

The following are the per-class plots that show sorted DSA values of each class in CIFAR-10.

{% include image.html url="./figures/RQ1_sorted_dsa_cifar_class.png" description="Sorted DSA values of adversarial examples for CIFAR-10 per class." %}

# RQ2: Does the selection of layers of neurons used for SA computation have any impact on how accurately SA reflects the behaviour of DL systems?

#### <span style="color:red">Figure 5' (NOT IN THE PAPER)</span>

This figure contains sorted LSA values from all layers in MNIST model. In the paper, pool1 was omitted.

{% include image.html url="./figures/RQ2_layer_selection_mnist_full.png" description="Sorted LSA of randomly selected 2,000 adversarial examples for MNIST from different layers." %}

#### <span style="color:red">Figure 5'' (NOT IN THE PAPER)</span>

This figure contains sorted LSA values from all layers in CIFAR-10 model. In the paper, only activation_1, activation_5, and activation_8 were presented.

{% include image.html url="./figures/RQ2_layer_selection_cifar_full.png" description="Sorted LSA of randomly selected 2,000 adversarial examples for CIFAR-10 from different layers." %}


# RQ3: Is SC correlated to existing coverage criteria for DL systems?

#### <span style="color:red">Figure 6' (NOT IN THE PAPER)</span>

This figure shows changes in various coverage criteria against increasing input diversity for each subject model. In the paper, only CIFAR-10 and Chauffeur were shown.

{% include image.html url="./figures/RQ3.png" description="Changes in various coverage criteria against increasing input diversity. We put additional inputs into the original test inputs and observe changes in coverage values." %}

#### <span style="color:red">Correlation Analysis for Figure 6' (NOT IN THE PAPER)</span>

In response to one of the reviewer questions, we have calculated Spearman's rank correlation coefficient between LSC/DSC and other coverage criteria. While the results show strong correlation, note that the sample sizes are very small (ranging from four to six) and some of the correlations are not statistically significant.

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;}
.tg th{font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;}
.tg .tg-name{text-align:left; vertical-align:middle;}
.tg .tg-name-vmiddle{text-align:center; vertical-align:middle;}
.tg .tg-name-center{text-align:center; vertical-align:middle;}
.tg .tg-num{text-align:right;vertical-align:top}
</style>
<div>
<table class="tg" style="margin: 0 auto;">
  <tr>
    <th class="tg-name-center" rowspan="2">DNN</th>
    <th class="tg-name-center" colspan="3">LSC</th>
    <th class="tg-name-center" colspan="3">DSC</th>
  </tr>
  <tr>
    <td class="tg-name">Criteria</td>
    <td class="tg-num">Spearman's \(\rho\)</td>
    <td class="tg-num">\(p\)--value</td>
    <td class="tg-name">Criteria</td>
    <td class="tg-num">Spearman's \(\rho\)</td>
    <td class="tg-num">\(p\)--value</td>
  </tr>
  <tr>
    <td class="tg-name-vmiddle" rowspan="4">MNIST</td>
    <td class="tg-name">NC</td>
    <td class="tg-num">0.926</td>
    <td class="tg-num">0.008</td>
    <td class="tg-name">NC</td>
    <td class="tg-num">0.926</td>
    <td class="tg-num">0.008</td>
  </tr>
  <tr>
    <td class="tg-name">KMNC</td>
    <td class="tg-num">1.000</td>
    <td class="tg-num">0.000</td>
    <td class="tg-name">KMNC</td>
    <td class="tg-num">1.000</td>
    <td class="tg-num">0.000</td>
  </tr>
  <tr>
    <td class="tg-name">NBC</td>
    <td class="tg-num">1.000</td>
    <td class="tg-num">0.000</td>
    <td class="tg-name">NBC</td>
    <td class="tg-num">1.000</td>
    <td class="tg-num">0.000</td>
  </tr>
  <tr>
    <td class="tg-name">SNAC</td>
    <td class="tg-num">0.971</td>
    <td class="tg-num">0.001</td>
    <td class="tg-name">SNAC</td>
    <td class="tg-num">0.971</td>
    <td class="tg-num">0.001</td>
  </tr>
  <tr>
    <td class="tg-name-vmiddle" rowspan="4">CIFAR-10</td>
    <td class="tg-name">NC</td>
    <td class="tg-num">0.941</td>
    <td class="tg-num">0.005</td>
    <td class="tg-name">NC</td>
    <td class="tg-num">0.941</td>
    <td class="tg-num">0.005</td>
  </tr>
  <tr>
    <td class="tg-name">KMNC</td>
    <td class="tg-num">1.000</td>
    <td class="tg-num">0.000</td>
    <td class="tg-name">KMNC</td>
    <td class="tg-num">1.000</td>
    <td class="tg-num">0.000</td>
  </tr>
  <tr>
    <td class="tg-name">NBC</td>
    <td class="tg-num">1.000</td>
    <td class="tg-num">0.000</td>
    <td class="tg-name">NBC</td>
    <td class="tg-num">1.000</td>
    <td class="tg-num">0.000</td>
  </tr>
  <tr>
    <td class="tg-name">SNAC</td>
    <td class="tg-num">1.000</td>
    <td class="tg-num">0.000</td>
    <td class="tg-name">SNAC</td>
    <td class="tg-num">1.000</td>
    <td class="tg-num">0.000</td>
  </tr>
  <tr>
    <td class="tg-name-vmiddle" rowspan="4">Dave-2</td>
    <td class="tg-name">NC</td>
    <td class="tg-num">0.949</td>
    <td class="tg-num">0.051</td>
    <td class="tg-name">NC</td>
    <td class="tg-num">N/A</td>
    <td class="tg-num">N/A</td>
  </tr>
  <tr>
    <td class="tg-name">KMNC</td>
    <td class="tg-num">0.949</td>
    <td class="tg-num">0.051</td>
    <td class="tg-name">KMNC</td>
    <td class="tg-num">N/A</td>
    <td class="tg-num">N/A</td>
  </tr>
  <tr>
    <td class="tg-name">NBC</td>
    <td class="tg-num">0.949</td>
    <td class="tg-num">0.051</td>
    <td class="tg-name">NBC</td>
    <td class="tg-num">N/A</td>
    <td class="tg-num">N/A</td>
  </tr>
  <tr>
    <td class="tg-name">SNAC</td>
    <td class="tg-num">0.949</td>
    <td class="tg-num">0.051</td>
    <td class="tg-name">SNAC</td>
    <td class="tg-num">N/A</td>
    <td class="tg-num">N/A</td>
  </tr>
  <tr>
    <td class="tg-name-vmiddle" rowspan="4">Chauffeur</td>
    <td class="tg-name">NC</td>
    <td class="tg-num">1.000</td>
    <td class="tg-num">0.000</td>
    <td class="tg-name">NC</td>
    <td class="tg-num">N/A</td>
    <td class="tg-num">N/A</td>
  </tr>
  <tr>
    <td class="tg-name">KMNC</td>
    <td class="tg-num">1.000</td>
    <td class="tg-num">0.000</td>
    <td class="tg-name">KMNC</td>
    <td class="tg-num">N/A</td>
    <td class="tg-num">N/A</td>
  </tr>
  <tr>
    <td class="tg-name">NBC</td>
    <td class="tg-num">1.000</td>
    <td class="tg-num">0.000</td>
    <td class="tg-name">NBC</td>
    <td class="tg-num">N/A</td>
    <td class="tg-num">N/A</td>
  </tr>
  <tr>
    <td class="tg-name">SNAC</td>
    <td class="tg-num">1.000</td>
    <td class="tg-num">0.000</td>
    <td class="tg-name">SNAC</td>
    <td class="tg-num">N/A</td>
    <td class="tg-num">N/A</td>
  </tr>
</table>
</div>