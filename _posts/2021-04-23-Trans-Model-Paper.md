---
layout: post
title: 知识表征学习必读论文
categories: 翻译模型
description: 翻译模型在知识图谱中的嵌入表示
keywords: 翻译模型,知识图谱,Embeddings,KRL,KE,知识表征学习,知识嵌入
---
# 总结性论文
* * 论文标题: Representation Learning: A Review and New Perspectives;
  * 作者:Yoshua Bengio, Aaron Courville, and Pascal Vincent. TPAMI 2013.
  * [论文地址](https://arxiv.org/pdf/1206.5538).
* * 论文标题:知识表示学习研究进展;
  * 作者:刘志远,孙茂松,林衍凯,谢若冰.计算机研究与发展2016.
  * [论文地址](http://crad.ict.ac.cn/CN/article/downloadArticleFile.do?attachType=PDF&id=3099)
* * 论文标题: A Review of Relational Machine Learning for Knowledge Graphs;
  * 作者:Maximilian Nickel, Kevin Murphy, Volker Tresp, Evgeniy Gabrilovich. Proceedings of the IEEE 2016.
  * [论文地址](https://arxiv.org/pdf/1503.00759.pdf)
* * 论文标题: Knowledge Graph Embedding: A Survey of Approaches and Applications.
  * 作者:Quan Wang, Zhendong Mao, Bin Wang, Li Guo. TKDE 2017.
  * [论文地址](http://ieeexplore.ieee.org/abstract/document/8047276/)

# 期刊和会议论文

* RESCAL模型
  * 论文标题: A Three-Way Model for Collective Learning on Multi-Relational Data.
  * 作者: Maximilian Nickel, Volker Tresp, Hans-Peter Kriegel. ICML 2011.
  * [论文地址](http://www.icml-2011.org/papers/438_icmlpaper.pdf),[论文代码](https://github.com/thunlp/OpenKE)
  * 描述:RESCAL模型是一种用于指示表示的张量分解方法,它能够分解的潜在组成部分执行集体学习.
* SE模型
  * 论文标题: Learning Structured Embeddings of Knowledge Bases.
  * 论文作者: Antoine Bordes, Jason Weston, Ronan Collobert, Yoshua Bengio. AAAI 2011.
  * [论文地址](http://www.aaai.org/ocs/index.php/AAAI/AAAI11/paper/download/3659/3898)
  * 描述:SE模型假设头实体和尾实体在依赖关系子控件中是相似的,其中每个关系是由两个不同的矩阵表示的.
* LFM模型
  * 论文标题:A Latent Factor Model for Highly Multi-relational Data.
  * 作者:Rodolphe Jenatton, Nicolas L. Roux, Antoine Bordes, Guillaume R. Obozinski. NIPS 2012.
  * [论文地址](http://papers.nips.cc/paper/4744-a-latent-factor-model-for-highly-multi-relational-data.pdf)
  * 描述:LFM基于双线性结构,该结构捕获数据交互作用的变异顺序,并在不同关系之间共享稀疏的潜在因素.
* NTN模型
  * 论文标题:Reasoning With Neural Tensor Networks for Knowledge Base Completion.
  * 作者:Richard Socher, Danqi Chen, Christopher D. Manning, Andrew Ng. NIPS 2013.
  * [论文地址](http://papers.nips.cc/paper/5028-reasoning-with-neural-tensor-networks-for-knowledge-base-completion.pdf)
  * 描述:NTN是一个神经网路,允许通过张量介导实体矢量的交互.NTN模型可能是迄今为止最具有表现力的模型,但是处理大型KG的方法还不够简单和有效
* TransE模型
  * 论文标题: Translating Embeddings for Modeling Multi-relational Data.
  * 作者: Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, Oksana Yakhnenko. NIPS 2013. paper code
  * [论文地址](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf),[论文代码](https://github.com/thunlp/OpenKE)
  * 描述: TransE模型是第一个介绍基于翻译的嵌入,其中将关系解释作为实体上的翻译操作.
* TransH模型
  * 论文标题: Knowledge Graph Embedding by Translating on Hyperplanes.
  * 作者: Zhen Wang, Jianwen Zhang, Jianlin Feng, Zheng Chen. AAAI 2014. paper code
  * [论文地址](http://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/viewFile/8531/8546),[论文源码](https://github.com/thunlp/OpenkE)
  * 描述:为了保留1-N,N-1,N-N关系的映射属性,TransH将关系解释为超平面上的平移操作,另外,TransH提出"Bern",这是一种构建负实例的策略.
* TransR & CTransR模型
  * 论文标题: Learning Entity and Relation Embeddings for Knowledge Graph Completion.
  * 作者: Yankai Lin, Zhiyuan Liu, Maosong Sun, Yang Liu, Xuan Zhu. AAAI 2015. 
  * [论文地址](http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9571/9523/),论文源码:[KB2E](https://github.com/thunlp/KB2E),[OpenKE](https://github.com/thunlp/OpenKE)
  * 描述: 一个实体可能会有多个方面,各种关系可能几种在实体的不同方面,TransR首先将实体从实体空间投影到对应的关系空间,然后在投影的实体之间构建转换.CTransR通过将不同的首位实体聚类,并为每个组学习不同的关系向量来扩展TransR,这是对每种关系类型内部相关性进行建模的初步探索.
* TransD模型
  * 论文标题:Knowledge Graph Embedding via Dynamic Mapping Matrix.
  * 作者:Guoliang Ji, Shizhu He, Liheng Xu, Kang Liu, Jun Zhao. ACL 2015.
  * [论文地址](http://anthology.aclweb.org/P/P15/P15-1067.pdf),论文源码[KB2E](https://github.com/thunlp/KB2E),[OpenKE](https://github.com/thunlp/OpenKE)
  * 描述:TransD 模型通过同事考虑实体和关系的多样性为每个实体关系对构造一个动态映射矩阵.与TransR/CTransR相比,TransD具有较少的参数,并且没有矩阵向量乘法.
* TransA模型
  * 论文标题:An Adaptive Approach for Knowledge Graph Embedding.
  * 作者:Han Xiao, Minlie Huang, Hao Yu, Xiaoyan Zhu. arXiv 2015.
  * [论文地址](https://arxiv.org/pdf/1509.05490.pdf)
  * 描述:应用椭圆等势超曲面并为关系加权特定的特征尺寸,TransA模型可以对复杂的实体和关系建模.
* KG2E模型
  * 论文标题: Learning to Represent Knowledge Graphs with Gaussian Embedding.
  * 论文作者: Shizhu He, Kang Liu, Guoliang Ji and Jun Zhao. CIKM 2015. paper code
  * 论文描述: 不同的实体和关系可能包含不同的确定性,这表示在给三元组评分的时候表示语义的置信度.KG2E通过高斯分布表示每个实体/关系,其中均值表示其位置,协方差表示其确定性.
  * [论文地址](https://pdfs.semanticscholar.org/941a/d7796cb67637f88db61e3d37a47ab3a45707.pdf),[论文源码](http://www.nlpr.ia.ac.cn/cip/~liukang/liukangPageFile/code/cikm15_he_code.zip)
* DistMult模型
  * 论文标题: Embedding Entities and Relations for Learning and Inference in Knowledge Bases.
  * 论文作者: Bishan Yang, Wen-tau Yih, Xiaodong He, Jianfeng Gao, Li Deng. ICLR 2015.
  * [论文地址](https://arxiv.org/pdf/1412.6575),[论文源码](https://github.com/thunlp/OpenKE)
  * 描述:DistMult模型基于双线性模型,其中每个关系由对角线而不是完整矩阵表示,DistMult模型享有与TransE模型相同的可伸缩性,与TransE模型相比,具有更加出色的性能.
* PTransE模型
  * 论文标题:Modeling Relation Paths for Representation Learning of Knowledge Bases.
  * 论文作者:Yankai Lin, Zhiyuan Liu, Huanbo Luan, Maosong Sun, Siwei Rao, Song Liu. EMNLP 2015.
  * [论文地址](https://arxiv.org/pdf/1506.00379.pdf),[论文源码](https://github.com/thunlp/KB2E)
  * 描述:多步关系路径在实体之间包含有丰富的推理模式.PtransE模型将关系路径视为实体之间的转换,并设计了一种出色的算法来测量关系路径的可靠性.实验表明,PTransE在KBC和RE任务方面取得了显著改进.
* RTransE
  * 论文标题: Composing Relationships with Translations.
  * 论文作者: Alberto García-Durán, Antoine Bordes, Nicolas Usunier. EMNLP 2015.
  * [论文地址](http://www.aclweb.org/anthology/D15-1034.pdf)
  * 描述:RTransE学习通过添加关系的相应翻译向量来显式地建模关系的构成.此外,实验还包括一个新的评估协议,其中该模型直接回答与关系构成有关的问题.
* ManifoldE模型
  * 论文标题: From One Point to A Manifold: Knowledge Graph Embedding For Precise Link Prediction.
  * 论文作者: Han Xiao, Minlie Huang and Xiaoyan Zhu. IJCAI 2016.
  * [论文地址](https://arxiv.org/pdf/1512.04792.pdf)
  * 描述:ManifoldE模型将基于评议原理的点式建模扩展为流形式建模,克服了几何形状过高的问题,并为精确链接预测实现了显著改进.
* TransG模型
  * 论文标题: A Generative Mixture Model for Knowledge Graph Embedding.
  * 论文作者: Han Xiao, Minlie Huang, Xiaoyan Zhu. ACL 2016.
  * [论文地址](http://www.aclweb.org/anthology/P16-1219),[论文源码](https://github.com/BookmanHan/Embedding)
  * 描述:知识图中的关系可能具有关联的实体对所揭示的不同含义.TransG通过贝叶斯非参数无限混合模型为关系生成多个平移分量.
* ComplEx模型
  * 论文标题: Complex Embeddings for Simple Link Prediction.
  * 论文作者: Théo Trouillon, Johannes Welbl, Sebastian Riedel, Éric Gaussier and Guillaume Bouchard. ICML 2016. paper code OpenKE
  * [论文地址](http://proceedings.mlr.press/v48/trouillon16.pdf),[论文源码](https://github.com/ttrouill/complex),[OpenKE](https://github.com/thunlp/OpenKE)
  * 描述:ComplEx模型通过引入复数值Embedding来扩展DistMult模型,以便更好地对非对称关系进行建模.事实证明,ComplEx模型将HolE归为特例.
* ComplEx extension模型        
  * 论文标题: Knowledge Graph Completion via Complex Tensor Factorization.
  * 论文作者: Théo Trouillon, Christopher R. Dance, Johannes Welbl, Sebastian Riedel, Éric Gaussier, Guillaume Bouchard. JMLR 2017.
  * [论文地址](https://arxiv.org/pdf/1702.06879.pdf),[论文源码](https://github.com/ttrouill/complex),[OpenKE](https://github.com/thunlp/OpenKE)
* HolE模型
  * 论文标题: Holographic Embeddings of Knowledge Graphs.
  * 论文作者: Maximilian Nickel, Lorenzo Rosasco, Tomaso A. Poggio. AAAI 2016.
  * [论文地址](http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12484/11828),[论文源码](https://github.com/mnick/holographic-embeddings),[OpenKE](https://github.com/thunlp/OpenKE)
  * 描述: HolE使用循环相关来创建成分表示.HolE可以捕捉丰富的交互,但同时仍然可以高效地进行计算.
* KR-EAR模型
  * 论文标题: Knowledge Representation Learning with Entities, Attributes and Relations.
  * 论文作者: Yankai Lin, Zhiyuan Liu, Maosong Sun. IJCAI 2016.
  * [论文地址](http://nlp.csai.tsinghua.edu.cn/~lyk/publications/ijcai2016_krear.pdf),[论文源码](https://github.com/thunlp/KR-EAR)
  * 描述:现有的KG关系可以分为属性和关系,它们具有相当不同的特征,KG-EAR是具有实体,属性和关系的KR模型法,它对实体描述之间的相关性进行编码.
* TranSparse模型
  * 论文标题: Knowledge Graph Completion with Adaptive Sparse Transfer Matrix.
  * 论文作者：Guoliang Ji, Kang Liu, Shizhu He, Jun Zhao. AAAI 2016.
  * [论文地址](http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/11982/11693),[论文源码](https://github.com/thunlp/Fast-TransX)
  * 描述:知识图谱中的实体和关系是异构的并且是不平衡的.为了克服异质性,TransSparse模型使用稀疏矩阵为关系建模.为了解决关系不平衡的问题,每个关系对于头实体和尾实体具有两个单独的稀疏传递矩阵.
* TKRL模型
  * 论文标题:Representation Learning of Knowledge Graphs with Hierarchical Types.
  * 论文作者:Ruobing Xie, Zhiyuan Liu, Maosong Sun. IJCAI 2016. 
  * [论文地址]http://www.thunlp.org/~lzy/publications/ijcai2016_tkrl.pdf),[论文源码](https://github.com/thunlp/TKRL)
  * 描述:实体应具有不同类型的多种表示形式,TKRL是捕获分层类型信息的首次尝试,这对于KRL具有重要意义.
* TEKE模型
  * 论文标题:Text-Enhanced Representation Learning for Knowledge Graph.
  * 论文作者:Zhigang Wang, Juan-Zi Li. IJCAI 2016.
  * [论文地址]()
  * 描述:TEKE合并了丰富的文本内容信息,以扩展知识图的语义结构.因此使得每个关系能够针对不同的头实体和尾实体拥有不同的表示,以便于更好地处理1-N,N-1,N-N关系.TEKE处理1-N,N-1,N-N关系的低性能和KG稀疏的问题.
* STransE模型
  * 论文标题:A Novel Embedding Model of Entities and Relationships in Knowledge Bases.
  * 论文作者:Dat Quoc Nguyen, Kairit Sirts, Lizhen Qu and Mark Johnson. NAACL-HLT 2016.
  * [论文地址](https://arxiv.org/pdf/1606.08140),[论文源码](https://github.com/datquocnguyen/STransE)
  * 描述:STransE是SE和TransE模型的简单组合,使用两个投影矩阵和一个转换向量来表示每个关系.STransE在链接预测评估上产生具有竞争力的结果.
* GAKE模型
  * 论文标题:Graph Aware Knowledge Embedding.
  * 论文作者:Jun Feng, Minlie Huang, Yang Yang, Xiaoyan Zhu. COLING 2016.
  * [论文地址](http://yangy.org/works/gake/gake-coling16.pdf),[论文源码](https://github.com/JuneFeng/GAKE)
  * 描述:将知识库作为有向图而不是独立的三元组,GAKE利用图上下文(邻居/路径/边缘上下文)来学习知识表示.此外,GAKE设计了一种注意力机制来学习不同主题代表的能力.
* DKRL模型
  * 论文标题:Representation Learning of Knowledge Graphs with Entity Descriptions.
  * 论文作者:Ruobing Xie, Zhiyuan Liu, Jia Jia, Huanbo Luan, Maosong Sun. AAAI 2016.
  * [论文地址](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12216/12004),[论文源码](https://github.com/thunlp/DKRL)
  * 描述:DKRL利用实体描述的优势来学习知识表示,zero-shot设置下的出色表现表明DKRL能够根据其描述为新颖的实体构建表示形式.
* ProPPR模型
  * 论文标题:Learning First-Order Logic Embeddings via Matrix Factorization.
  * 论文作者:William Yang Wang, William W. Cohen. IJCAI 2016.
  * [论文地址](https://www.cs.ucsb.edu/~william/papers/ijcai2016.pdf)
  * 描述:ProPPR模型是第一个研究从头开始学习低维一阶逻辑embeddings,同时将基于公式embeddings的概率逻辑推理扩展到大型知识图的问题的形式研究.
* SSP模型
  * 论文标题: Semantic Space Projection for Knowledge Graph Embedding with Text Descriptions.
  * 论文作者: Han Xiao, Minlie Huang, Lian Meng, Xiaoyan Zhu. AAAI 2017.
  * [论文地址](http://www.aaai.org/Conferences/AAAI/2017/PreliminaryPapers/14-XiaoH-14306.pdf)
  * 描述:SSP模型通过针对最新技术水平的基线进行embedding改进来对三元组和文本相关性之间的强相关进行建模.
ProjE模型
  * 论文标题: Embedding Projection for Knowledge Graph Completion.
  * 论文作者: Baoxu Shi, Tim Weninger. AAAI 2017.
  * [论文地址](http://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/download/14279/13906),[论文源码](https://github.com/bxshi/ProjE)
  * 描述: ProjE将KGC任务是为排名问题,并将候选实体投影到一个向量上,该向量表示输入三元组已知部分的组合嵌入.此外,ProjE可以优化候选实体列表的整体排名损失,可以将ProjE是为NTN的简化版本.
