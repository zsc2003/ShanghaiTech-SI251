# Introduction
现代深度学习模型由有向图中的参数化处理节点（或层）组成。有一整套不同的模型架构，主要按图结构和参数共享机制分类[33]。在所有情况下，处理节点从输入到输出的数据转换功能都是明确定义的。然后，通过沿图中的边反向传播误差信号和调整参数以最小化误差，实现端到端的学习。几乎普遍的做法是，将误差信号编码为某个全局目标（或正则化损失）函数的梯度，并使用基于随机梯度下降的方法迭代更新参数。自动微分法（或手工梯度法）用于计算处理节点输出相对于输入的导数，然后结合微分链规则在图中向后进行微分。这样，当信号向后通过每个处理节点时，就能有效计算出模型所有参数的误差梯度。

在本文中，我们提倡一种新的端到端可学习模型，我们称之为深度声明式网络（DDNs），它将最近关于可微分优化的一些研究成果[1, 2, 3, 4, 28]和相关想法[11, 45, 47]整合到一个框架中。DDN 中的节点不是显式定义前向处理函数，而是通过指定行为来隐式定义。也就是说，节点的输入输出关系是根据数学优化问题中的目标和约束条件来定义的，输出是以输入（和参数）为条件的问题解决方案。重要的是，正如我们将要展示的，通过隐式微分，我们仍然可以通过 DDN 进行反向传播。
此外，梯度计算不需要了解用于解决优化问题的方法，只需要了解目标和约束条件的形式，因此在前向传递过程中可以使用任何最先进的求解器。

DDN 超越了传统的深度学习模型，因为任何明确定义的前向处理函数都可以被定义为 DDN 中的一个节点。此外，声明定义的节点和明确定义的节点可以共存于同一个端到端可学习模型中。为了明确区分，当两种类型的节点出现在同一模型中时，我们将前者称为声明式节点，后者称为命令式节点。为此，我们在 PyTorch [39]（一种流行的深度学习软件库）中开发了一种 DDN 参考实现，同时支持声明式节点和命令式节点。

我们提出了一些理论结果，说明了可以计算精确梯度的条件以及这种梯度的形式。我们还讨论了无法计算精确梯度的情况（如非光滑目标），但仍然可以找到下降方向，从而对模型参数进行随机优化。我们通过一系列示例探讨了这些想法，并分别使用改进的 ResNet [30] 和 PointNet [40] 架构在图像和点云分类问题上进行了实验测试。

深度神经网络声明式观点的决定性优势在于，它可以将经典的有约束和无约束优化算法作为一个更大的、端到端的可学习网络中的模块化组件来使用。这就扩展了神经网络层的概念，例如包括几何模型拟合，如相对或绝对姿态求解器或捆绑调整、模型预测控制算法、期望最大化、匹配、最优传输和结构预测求解器等等。此外，视角的改变还能帮助我们设想具有更理想特性的标准神经网络操作的变体，例如鲁棒特征池而不是标准平均池。通过将局部模型拟合作为更大模型中的一个组成部分，还有可能减少网络中的不透明和冗余。例如，我们可以直接使用（通常是非线性的）基础物理和数学模型，而不必在网络中重新学习这些模型。重要的是，这使我们能够对模型内的表征提供保证和强制执行硬约束（例如，几何模型内的旋转是有效的，或排列矩阵的归一化是正确的）。此外，这种方法在不存在闭合形式解的情况下仍然适用，允许在内部使用具有无差别步骤的复杂方法（如 RANSAC [26]）。即使在这种情况下，网络参数的全局端到端学习仍然是可能的。

由于这是一种新方法，仍有一些挑战有待解决。我们将在本文中讨论其中的一些挑战，但还有许多挑战需要我们和社区在今后的工作中加以解决。与声明式网络有关的一些初步想法出现在以前的著作中（下文将讨论），但据我们所知，本文是第一篇提出描述这些模型的一般连贯框架的论文。

# Background and related work
通过声明节点进行微分的能力依赖于隐函数定理，该定理由来已久，其根源可追溯到笛卡尔、莱布尼兹、伯努利和欧拉的著作[44]。柯西（Cauchy）首次将该定理置于严格的数学基础之上，而迪尼（Dini）则首次以现代多元形式提出了该定理[21, 32]。粗略地说，该定理说明了对于隐含定义的函数 $f(x, y) = 0$ ，变量 y 相对于另一变量 x 的导数存在的条件，并提供了在导数确实存在时计算 y 相对于 x 的导数的方法。

在深度学习中，（声明式）节点的输出相对于输入的导数有助于通过反向传播进行端到端的参数学习 [33, 41]。从这个意义上说，学习问题被表述为给定误差度量或正则化损失函数的优化问题。当网络中出现声明节点时，计算网络输出以及损失函数本身就需要解决一个内部优化问题。从形式上看，我们可以将学习问题视为上层优化问题，而网络输出则是在双层优化框架内通过下层优化问题获得的[7, 46]。

双层优化（以及对隐式微分的需求）已出现在机器学习文献的各种环境中，其中最著名的是金属学习问题。例如，Do 等人[19] 将确定对数线性模型正则化参数的问题视为双层优化问题。Domke [20] 解决了基于连续能量模型的参数学习问题，其中推理（相当于神经网络中的前向传递）需要找到所谓能量函数的最小值。由此产生的学习问题必然是双水平的。最后，Klatzer 和 Pock [31] 提出了一个双层优化框架，用于选择支持向量机的超参数设置，从而避免了交叉验证的需要。

在计算机视觉和图像处理领域，双层优化已被用于制定像素标记问题的解决方案。Samuel 和 Tappen [42] 提出用双层优化法学习连续马尔可夫随机场的参数，并将其技术应用于图像去噪和内绘制。该方法是基于能量的模型学习[20]的一个特例。Ochs 等人[37] 将双层优化扩展到处理非平滑的低层次问题，并将其方法应用于图像分割任务。

最近的研究开始考虑将特定优化问题置于深度学习模型中的问题 [4, 11, 28, 45, 47]。可以认为，这些方法通过开发特定的声明式组件，为 DDN 奠定了基础。Gould 等人[28]总结了一些区分无约束、线性约束和不等式约束优化问题的一般结果。对于无约束和相等约束问题，这些结果是精确的，而对于不等式约束问题，他们则使用了内点近似法。我们将这些结果扩展到非线性平等和不平等约束问题的精确微分情况。

Amos和Kolter[4]还针对二次型程序（QPs）的特殊情况，展示了如何通过优化问题进行微分。Amos [3]全面介绍了这项工作，包括对更一般的锥形程序的讨论。同样，Agrawal 等人[2]报告了通过具有数百万个参数的锥形程序进行高效微分的结果。在这两种情况下（二次方程式程序和锥形程序），问题都是凸的，并且存在高效的算法来找到最小值。我们对声明性节点的凸性不做限制，但仍假定存在一种在前向传递中对它们进行评估的算法。此外，我们的论文还建立了一个统一的框架，以便在端到端可学习模型--深度声明网络（DDN）中查看这些工作。

其他研究还考虑了通过离散子模块问题进行微分的问题 [18, 45]。这些问题有一个很好的特性，即凸松弛最小化仍能得到子模块最小化问题的最优解，从而可以计算导数 [18]。对于亚模态最大化问题，存在着找到近似解的多项式时间算法。Tschiatschek 等人[45]的研究表明，对这些解进行平滑处理可得到可微分模型。

最近提出的 SATNet [47]，在精神上与深度声明式网络的理念非常接近。在这里，MAXSAT 问题是通过求解可微分的半定式程序 (SDP) 来逼近的。由于采用了隐式微分，因此无需显式展开优化过程，也就无需存储雅各比，从而在后向过程中节省了内存和时间。该方法被应用于解决以图像形式呈现的数独问题，这就要求学习网络对逻辑约束进行编码。

适合深度声明式网络框架的另一类有趣的模型是 Bai 等人最近提出的深度平衡模型[6]。在这里，模型执行一系列定点迭代 $y^(t) = f(x, y^(t-1))$，直到在前向传递中收敛。Bai 等人[6]的研究表明，与其通过无限制的定点迭代序列进行反向传播，不如通过隐函数定理直接计算解 y 相对于输入 x 的导数，即观察到 y 满足隐函数 $f(x, y) - y = 0$。

Chen 等人[11]展示了如何利用邻接灵敏度方法对某个时间 T 的常微分方程（ODE）初值问题进行微分，并将残差网络[30]视为此类问题的离散化。这种方法可以解释为用积分约束函数求解可行性问题，因此是一种特殊的声明式节点。通过求解第二个增强 ODE，可以优雅地计算后向传递中的必要梯度。虽然这类问题也可以包含在我们的声明式框架中，但在本文中，我们将重点讨论可以用封闭形式表达的两次可变目标函数和约束函数的结果。

还有一些研究提出了基于优化问题的可微分组件，以解决特定任务。在视频分类方面，Fernando和Gould[23, 24]展示了如何通过深度学习模型中的秩池算子[25]进行区分，这涉及到解决支持向量回归问题。这种方法随后被推广到视频的子空间表示和流形学习中[12]。

Santa Cruz 等人[43]提出了一种深度学习模型，用于学习视觉属性排序和比较的置换矩阵。他们提出了两种变体，都将置换矩阵表示法放宽为双随机矩阵表示法。第一种变体涉及对行和列进行迭代归一化，以将正矩阵近似投射到双随机矩阵集合上。第二个变式是将投影表述为二次问题并精确求解。

Lee 等人[34]考虑了视觉识别中的少量学习问题。他们将可变 QP [4] 嵌入到深度学习模型中，允许训练线性分类器，作为泛化到新视觉类别的基础。他们在标准基准上取得了可喜的成果，并报告了较低的训练开销，即解决 QP 所需的时间与图像特征提取的时间大致相同。

在规划和控制方面，Amos 等人[5] 提出了一种强化学习环境下的可微分模型预测控制器。他们在经典的钟摆和车杆问题上表现出了卓越的性能。De Avila Belbute-Peres 等人[14]展示了如何通过物理模型进行微分，特别是线性互补问题的最优解，从而将模拟物理环境置于端到端可学习系统中。

最后，M ´arquez-Neila等人[35]研究了使用克雷洛夫子空间方法对深度神经网络的输出施加硬约束。该方法被应用于人体姿态估计任务，并证明了训练一个强制硬约束的高维模型的可行性，尽管与软约束基线相比没有改进。

# Notation
我们的结果要求对向量参数进行向量值函数微分。为了便于演示，我们在此澄清一下我们的符号。考虑函数 $f(x,y,z)$，其中 y 和 z 本身都是 x 的函数。我们有：$$\frac{d}{dx}f = \frac{\partial f}{\partial x} \frac{dx}{dx} + \frac{\partial f}{\partial y} \frac{dy}{dx} + \frac{\partial f}{\partial z} \frac{dz}{dx}$$ 根据微分的链式法则。

对于取向量参数的函数 $f：\mathbf{R^n} → \mathbf{R}$，我们将导数向量写为 
![1.png](/image/1.png)

对于向量值函数 $f ：\mathbf{R} → \mathbf{R^m}$，我们写为
![2.png](/image/2.png)

更一般地说，我们定义 f 的导数 $Df ：\mathbf{R^n} → \mathbf{R^m} $ 的 m × n 矩阵，其条目为
![3.png](/image/3.png)

那么，h(x) = g(f(x)) 的链式法则就是
![4.png](/image/4.png)

其中的矩阵自动具有正确的维数，并适用标准的矩阵-矢量乘法。在求偏导数时，我们使用下标来表示计算导数的形式变量（其余变量固定不变），例如 $D_Xf(x,y)$。为简洁起见，我们使用速记 $D^{2}_{XY} f$ 表示 $D_X(D_Y f)^{T}$。

当多变量函数没有下标时，我们将 D 视为相对于自变量的总导数。因此，以 x 为自变量，方程 1 的向量版本变为
![5.png](/image/5.png)