---
layout: post
title: 翻译模型在知识图谱中的应用(一):基本模型
categories: 翻译模型
description: 翻译模型在知识图谱中的嵌入表示
keywords: 翻译模型,知识图谱,Embeddings
---
# 1. 翻译模型简介
&ensp;&ensp;&ensp;&ensp;将实体向量表示(Embedding)在低维稠密向量空间中，然后进行计算和推理.应用的主要方向有三元组分类、链接预测等问题.具体的一些翻译模型主要有:基本翻译模型(TransE, TransH, TransR, etc)、双线性(RESCAL, DisMult, ComplEx, etc)、双曲几何(Poincare, MuRE, etc)、神经网络(ConvE, CapsE, etc)、旋转(RotatE, QuatE, DihEdral, etc)等等类别,近几年对于翻译模型有很多不同的变种,在接下来几篇博文中会详细梳理这些翻译模型.

给定训练集$S$,由包含头实体和尾实体以及他们之间关系的三元组(h,r,t)构成,通过模型学习实体和关系的嵌入向量.模型的基本思想是,由r边产生的函数关系,这个关系映射嵌入的转换,也就是说,在学习过程中希望$h+r$与尾实体$t$尽可能相似.为了学习这种嵌入表示,对训练集构建基于边缘的"最小化评价指标函数":

$$\mathcal{L}=\sum\limits_{(h,r,t)\in{S}}\sum\limits_{(h^{\prime},r,t^{\prime})\in{S^{\prime}}}\left[\gamma+d(h+r,t)-d(h^{\prime}+r,t^{\prime})\right]_{+}$$

&ensp;&ensp;&ensp;&ensp;其中$[\cdot]_{+}$表示$\max(0,x)$,$d(\cdot)$表示一种衡量三元组能量的距离函数,可以是L1范数或者是L2范数,r表示的是三元组正实例和负实例的间隔,为超参数.从姆比爱函数中需要负实例$(h^{\prime},r,t^{\prime})$来进行函数的优化.知识图谱中存储的均为正实例,所以在这里需要人为地构建负实例.常常使用以下的方法构建负实例:从知识图谱中随机抽取其他头实体和尾实体来替换当前三元组中头实体或者是尾实体,但是为了防止替换之后的三元组也是正实例,这里算法的限制是替换过程不能同时进行.


$$S_{(h,r,t)}^{\prime}=\left\{(h^{\prime},r,t)\left|h^{\prime}\in{E}\right.\right\}\cup\left\{(h,r,t^{\prime})\left|t^{\prime}\in{E}\right.\right\}$$

然后,在可能的$(h,r,t)$三元组基础上,使用随机梯度下降算法进行优化,并且添加附加的约束条件:\\

+ 实体的嵌入L2范数为1
+ 标签Embedding的没有正则化或范数约束,这是因为防止了训练过程认为增加实体Embedding规范来微不足道地最小化L.

有时候也会应用到cos距离模型等等来计算相对应的分数.

# 2. 基本翻译模型
## 2.1 TransE模型
&ensp;&ensp;&ensp;&ensp;最初在知识图谱中提出的一种模型是TransE<sup>1</sup>模型.TransE模型最初的目的是将实体间的关系表示为在实体的低维空间中的平移等转换操作.尽管是一种简单的空间转换操作,但是也是一种很强大的模型.Multi-relation data(多元关系数据)指的是在有向图中,节点对应的是entities(实体)和edges(关系),给定知识图谱(h,r,t),其中h为头实体集合,t为尾实体集合,r是关系集合.在很多领域,例如社会网络分析(分析用户实体的社交关系),推荐系统(用户实体和商品实体间的购买、评级、浏览和搜索的关系),知识图谱.现在提出的问题和任务是,使用多关系数据构建一个模型,能够提供一种有效的工具,自动添加新的事实而无需通过额外的知识.

(1) 构建多元关系数据信息:通常关系数据包括有单一关系数据和多元关系数据.单一关系通常是结构化的,可以直接进行简单的推理;多元关系则依赖于多种类型的实体和关系,因此需要一种通用的方法能够同时考虑异构关系.

(2) 关系可以作为embedding空间的翻译.TransE模型中关系则是表示为Embedding空间中的平移.即如果存在头实体和尾实体之间的关系实体对(h,r,t),那么尾实体t必须接近于头实体在地位嵌入空间中添加一个关系向量r的结果.

&ensp;&ensp;&ensp;&ensp;实际上TransE模型就是直接计算出$d(h+r,t)=\left|\left|(h+r)-t\right|\right|\approx{0}$来对实体和关系建模,将它们映射到相同的向量空间中,如图所示.
<div>
<p style="text-align:center;"><img src="/images/posts/TransModel/TransE.png" width="35%" alt="TransE模型表示图" /></p>
</div>
## 2.2 TransH模型
&ensp;&ensp;&ensp;&ensp;TransE是一种有效的方法,同时能够获得比较好的预测结果,虽然可以在大量数据的条件下可以快速有效进行训练,但是模型过于简单,并不能够很好的表征实体对之间的语义关系.另外对于处理负载关系数据(一对多、多对一和多对多)时候,过于简单的假设可能导致错误的表征,尤其是对于一对多的情形,即同一个头实体$h$和关系$r$可能对应多个尾实体,这样训练之后的结果可能是$h_{1}\approx{h_{2}}\approx{h_{3}}$,这明显是错误的,需要对此进一步进行改进.这样TransH模型就被提出来了.知识图谱是一种多元关系表示的图,其中包含有一些实体(节点)和关系(不同类型的边)组成.一个边的示例代表一个事实,包含头实体、关系和尾实体,通常表示为$(h,r,t)$.知识图谱依然存在以下的两个主要的挑战:

(1) 知识图谱是一种由符号和逻辑组成的系统,基于此的一些应用涉及到连续空间内大规模的计算

(2) 很难把全局信息聚集在图中.

&ensp;&ensp;&ensp;&ensp;TransE总而言之有以下的一些缺点

+ TransE模型中无法解决自反,1-N,N-N,N-1等等关系.
+ TransE中构造负三元组样本的方式是随机替换三元组中的h或者是t,由于只是图谱不能包含所有的知识,可能会构造出正确的三元组,但是却将它当做负样本进行处理,这样就构造出了假阴性的标签.

&ensp;&ensp;&ensp;&ensp;TransH由(Zhang J .)<sup>2</sup>等提出,传统的方法是基于形式化逻辑推理的方法,但是处理长范围的推理显得不那么容易处理,健壮性较低.TransH模型是将实体映射到关系所在的超平面上,每个关系包含两个向量$w_{r}$和$d_{r}$,分别表示超平面的法向量和在超平面内的翻译向量.关系对应的超平面确定,因此法向量$w_{r}$也是确定的,而三元组中头实体和尾实体分别投影到这个超平面上形成的向量之间存在一定的映射关系,$d_{r}$便是这个映射关系的翻译向量.

假设一个三元组(h,r,t)对应的向量分别是h,r,t,关系r的对应投影矩阵设为$w_{r}$,如图所示

<div>
<p style="text-align:center;"><img src="/images/posts/TransModel/TransH.png" width="35%" alt="TransH模型向量图" /></p>
</div>

&ensp;&ensp;&ensp;&ensp;总之,TransH中有以下的一些基本思想:

(1) 提出一种基于翻译的模型TransH,将relation看作是在低维空间中由head到tail在某个超平面上的一种翻译;

(2) 提出了一种构造负三元组样本的方法,为每种替换设置不同的概率,使得1-n关系中替换掉h和n-1关系中替换掉t有着更大的概率.

&ensp;&ensp;&ensp;&ensp;TranH模型方法如下所示:

$$h_{\perp}=h-w_{r}^{T}hw_{r}$$

$$t_{\perp}=t-w_{r}^{T}tw_{r}$$
        
所以这样的三元组评判的方法为$d(h+r,t)=\left\|\left\|h_{\perp}+d_{r}-t_{\perp}\right\|\right\|$.

&ensp;&ensp;&ensp;&ensp;为保证约束条件:

$$\forall{e\in{E}},||e||_{2}\leq{1}$$

$$\forall{r\in{R}},\dfrac{|w_{r}^{T}d_{r}|}{||d_{r}||_{2}}\leq{\epsilon}$$

$$\forall{r\in{R}},||w_{r}||_{2}=1$$

所以在margin loss函数中加入以下形式的正则化项

$$\mathcal{L}=\sum\limits_{(h,r,t)\in{S}}\sum\limits_{(h^{\prime},r,t^{\prime})\in{S^{\prime}}}\left[\gamma+d(h+r,t)-d(h^{\prime}+r,t^{\prime})\right]_{+}$$

$$+C\left\{\sum\limits_{e\in{E}}\left[||e||_{2}^{2}-1\right]_{+}+\sum\limits_{r\in{R}}\left[\dfrac{(w_{r}^{T}d_{r})^{2}}{||d_{r}||_{2}^{2}}-\epsilon^{2}\right]_{+}\right\}$$

$C$是一个衡量约束项的重要性的一个超参数.设每个tail对应的head数量的平均数为tph,每个head对应的tail数量的平均数为hpt,定义参数为$\dfrac{\text{tph}}{\text{tph}+\text{hpt}}$和$\dfrac{\text{hpt}}{\text{tph}+\text{hpt}}$的二项分布来进行抽样,即有以下的方法:

(1) 以$\dfrac{\text{tph}}{\text{tph}+\text{hpt}}$的概率来替换头实体

(2) 以$\dfrac{\text{hpt}}{\text{tph}+\text{hpt}}$的概率来替换尾实体

## 2.3 TransR模型
&ensp;&ensp;&ensp;&ensp;TransH模型能够通过将关系视为一种从头实体到尾实体的翻译机制来获得实体和关系的表征,然而一个实体可能会有多个不同方面的特征,关系可能关注实体不同方面的特征,公共的实体特征不能够表征.所以提出TransR模型<sup>3</sup>，即构建实体和关系表征,将实体空间和关系空间相分离.训练的时候首先通过将实体映射到关系空间中,其次在两个投影实体之间构建翻译关系.

&ensp;&ensp;&ensp;&ensp;TransR模型的主要思路如下所示,如图:

<div>
<p style="text-align:center;"><img src="/images/posts/TransModel/TransR.png" width="50%" alt="TransH模型向量图" /></p>
</div>

&ensp;&ensp;&ensp;&ensp;假设实体对(h,r,t),首先根据当前的关系r将头尾实体分别映射到关系空间中$h_{r},t_{r}$,然后在关系空间中建模$h_{r}+r\approx{t_{r}}$.另外在特定的关系情况下,实体对通常表现出不同的模式,因而不能单纯地将关系直接与实体对进行操作,通过将不同的头尾实体对聚类成组,并为每个组学习不同的关系向量来扩展TransR模型,这种模型在论文中称为CTransR模型.

&ensp;&ensp;&ensp;&ensp;对于TransR模型来说,假设三元组实体对为$(h,r,t)$的表征为$h,t\in{R^{k}},r\in{R^{d}}$,其中$k\neq{d}$,对于每个关系$r$给定映射矩阵$M\in{R^{k\times{d}}}$,所以这样的转换可以得到

$$h_{r}=hM_{r},t_{r}=tM_{r}$$

这样得分函数定义为$d(h+r,t)=\left\|\left\|h_{r}+r-t_{r}\right\|\right\|_{2}^{2}$,其中的约束条件为

$$\left|\left|h\right|\right|_{2}\leq{1},\left|\left|r\right|\right|_{2}\leq{1},\left|\left|t\right|\right|_{2}\leq{1},\left|\left|hM_{r}\right|\right|_{2}\leq{1},\left|\left|tM_{r}\right|\right|_{2}\leq{1}$$

&ensp;&ensp;&ensp;&ensp;对于CTransR模型来说,计算的方法如下所示

1.  (聚类操作)首先将输入示例分为多个组,对于特定的关系$r$,所有实体对$(h,t)$可以被聚类到多个簇中,每个簇的实体对(h,r,t)可以被聚类到多个簇中,每个簇中的实体对可以被认为与关系$r$有关系.
2. 为每个簇对应的关系向量$r_{c}$表征,并得到$M_{r}$,然后将每个簇中的头实体和尾实体映射到对应关系空间中

$$h_{r,c}=hM_{r},t_{t,c}=tM_{r}$$

最后得分函数如下所示:

$$d(h+r,t)=\left|\left|h_{r,c}+r_{c}-t_{r,c}\right|\right|_{2}^{2}+\alpha\left|\left|r_{c}-r\right|\right|_{2}^{2}$$

## 2.4 TransD模型
&ensp;&ensp;&ensp;&ensp;TransD模型是由文章<sup>4</sup>所提出来的一个模型,对于TransE、TransH和TransR模型来说,认为头实体到尾实体可以被认为是一种翻译模型,TransD模型则更为细粒度的一个模型,相比之前的模型有所提高,TransD模型中使用两个embedding表征实体之间的关系,第一个向量表征实体关系,另一个是用来构建动态映射矩阵.TransR模型具有以下的一些缺点:

(1) 对于特定的关系r,所有实体共享同一个语义空间$M_{r}$,因此实体需要映射待不同的语义空间中;

(2) 实体和关系的投影操作是一个连续迭代的操作,仅仅依靠关系进行推理是不足的;

(3) 矩阵向量带来大量的参数运算量.

&ensp;&ensp;&ensp;&ensp;TransD模型如图所示
<div>
<p style="text-align:center;"><img src="/images/posts/TransModel/TransD.png" width="60%" alt="TransH模型向量图" /></p>
</div>

&ensp;&ensp;&ensp;&ensp;定义了两个向量,第一个向量表征实体或者关系的语义,另外一个向量(投影向量)表示如何将实体从实体空间映射到关系空间中,因此每个实体对有唯一的矩阵.矩阵$M_{rh},M_{rt}$分别是实体$h,t$的映射矩阵.这样就会得到

$$M_{rh}=r_{p}h_{p}^{T}+I^{m\times{n}},M_{rt}=r_{p}t_{p}^{T}+I^{m\times{n}}$$

$$h_{\perp}=M_{rh}h,t_{\perp}=M_{rt}t$$

所以最后的评分函数表示为

$$d(h+r,t)=\left|\left|h_{\perp}+r-t_{\perp}\right|\right|_{2}^{2}$$

## 2.5 Transparse模型
&ensp;&ensp;&ensp;&ensp;TransE,TransH,TransR(CTransR)和TransD模型均一步步改进了知识表示的方法,完善知识补全工作上逐渐提高效果.这些模型中忽略了知识图谱中的两个重要特性

(1) 异质性:知识图谱中的异质性是指不同关系对应的实体对数量不一致

(2) 不平衡性:是指头尾实体的数量是不一致的
&ensp;&ensp;&ensp;&ensp;由于数量的不对等,所以这样数量较多的对应关系的实体对或者头尾实体它们包含的信息应该越多,而前面的几种模型忽略了这一点,使得针对每个实体对都用同样的方法训练,势必会导致数量多的部分出现欠拟合,数量少的部分出现过拟合现象,所以由此提出TranSparse模型来改进这一个问题.

&ensp;&ensp;&ensp;&ensp;解决这样一个问题的策略是引用系数矩阵,首先对于异质性,提出了TranSparse(Share),系数因子取决于关系链接对应的实体对数量,并且两个实体对应的关系投影矩阵是相同的.对于不平衡性,提出TranSparse(Separate),每个关系对应的实体对中,头尾实体使用不同的关系投影矩阵.

&ensp;&ensp;&ensp;&ensp;稀疏矩阵指的是一个矩阵中包含有大量的零元素,而零元素所占重元素个数的比值为稀疏因子$\theta$,稀疏因子$\theta$越大表示这个矩阵是越稀疏的,用$M(\theta)$表示系数因子为$\theta$的矩阵.

&ensp;&ensp;&ensp;&ensp;主要的思想:先前的模型中,不论关系对应的实体或者实体对数量多少,训练参数是相同的,因此可能导致数量少的实体或者实体对训练会过拟合,数量多的实体或者实体对训练欠拟合,故而这需要考虑到参数与实体对之间的数量关系.在TranSparse中,假设$N_{r}$表示关系$r$链接的实体对数量,$N_{r^{\*}}$表示其中最大值,$r^{\*}$表示对应的关系,再设$\theta_{\min}(0\leq{\theta_{\min}}\leq{1})$表示的是矩阵$M_{r^{*}}$的稀疏因子,则会有

$$\theta_{r}=1-(1-\theta_{\min})\dfrac{N_{r}}{N_{r*}}$$

&ensp;&ensp;&ensp;&ensp;通过此公式可知最大实体对数量为基数,其他实体对数量与之比值作为相对复杂度,该公式可计算对应关系投影矩阵的系数因子,其次可以将头尾实体分别映射到关系空间中

$$h_{p}=M_{r}(\theta_{r})h,t_{p}=M_{r}(\theta_{r})t$$

&ensp;&ensp;&ensp;&ensp;TranSparse(Separate)与Share不同,头尾实体分别映射到不同的关系空间中.$N_{r}^{l}$表示"头实体-关系"映射矩阵$M_{r}^{h}(\theta_{r}^{h})$和"尾实体-关系"映射矩阵$M_{r}^{t}(\theta_{r}^{t})$.对于关系$r$,最大数量头尾实体$h^{\*}$和$t^{\*}$分别对应的数量为$N_{r^{\*}},N_{t^{\*}}$.因此"头实体-关系"映射矩阵的稀疏因子为

$$\theta_{r}^{h}=1-(1-\theta_{\min})\dfrac{N_{r}^{h}}{N_{r*}^{h^{*}}}$$

故而头尾实体分别映射到关系空间中:

$$h_{p}=M_{r}^{h}(\theta_{r}^{h})h,t_{p}=M_{r}^{t}(\theta_{r}^{t})t$$

最后得分函数为

$$d(h,r,t)=\left|\left|h_{p}+r-t_{p}\right|\right|^{2}_{l1/2}$$

# 3.模型评价方法

&ensp;&ensp;&ensp;&ensp;翻译模型经常使用的到的算法评价指标有以下的几种:

(1) 正确实体的平均排名.正确实体的平均排序得分简称为MeanRank,此值越小越好,这也是衡量链接预测的重要指标.
(2) 正确实体排名在前10的概率,正确实体排在前10名的概率简称为Hits\@10,此值越大越好,这也是衡量链接预测的重要指标.
(3) 准确率,三元组分类任务使用准确率作为评价指标,计算方法为

$$ACC=\dfrac{T_{p}+T_{n}}{N_{pos}+N_{neg}}$$

&ensp;&ensp;&ensp;&ensp;其中,$T_{p}$表示预测正确的正例三元组个数;$T_{n}$表示预测正确的负例三元组个数;$N_{pos}$和$N_{neg}$分别表示训练集中正例三元组和负例三元组个数.ACC越高,表示模型在三元组分类这一任务上的效果是越好的.

&ensp;&ensp;&ensp;&ensp;为了科学、一致地评价各类Embedding表示算法的性能,需要使用标准的实体关系数据集进行测试和对比.目前常用的实体关系数据集进行测试和对比.目前尝试用的实体关系数据集有以下的几种

(1) WN18,它是WordNet知识库的一个子集,有关系18个,实体40943个;

(2) FB15K,FreeBase中一个相对稠密的子集,有关系1345个,实体14951个;

(3) WN11,是WordNet知识库的一个子集,有关系11个,实体38696个;

(4) FB13,是FreeBase中的一个相对稠密的子集,有关系13个,实体75043个;

(5) FB40K,是FreeBase中一个相对稠密的子集,有关系11个1336个,实体39528个;

(6) MPBC\_20,有关系20个,实体175624个;

(7) FB15K-237,是FreeBase中的一个子集,有关系237个,实体14541个.

# 参考文献
[1] TransE模型:Antoine Bordes,Nicolas Usunier,Alberto Garcia-Duran, Jason Weston, and OksanaYakhnenko.Translating embeddings for modelingmulti-relational data. InNIPS, pages 2787–2795, 2013.

[2] TransH模型:Zhang J . Knowledge Graph Embedding by Translating on Hyperplanes[J]. AAAI - Association for the Advancement of Artificial Intelligence, 2015.

[3] TransR模型:Lin Y, Liu Z, Zhu X, et al. Learning Entity and Relation Embeddings for Knowledge Graph Completion. AAAI. 2015.

[4] TransD模型:Ji G , He S , Xu L , et al. Knowledge Graph Embedding via Dynamic Mapping Matrix[C]// Meeting of the Association for Computational Linguistics \& the International Joint Conference on Natural Language Processing. 2015.

[5] Transparse模型Ji G, Liu K, He S, et al. Knowledge Graph Completion with Adaptive Sparse Transfer Matrix. AAAI. 2016.

