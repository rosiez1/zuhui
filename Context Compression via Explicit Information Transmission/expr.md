=== ComprExIT latent message ===
context tensor        : (1, 512)
raw compressed Z      : (1, 128, 2048)
decoder-space Z       : (1, 128, 2048)
valid context tokens  : 51
valid compression slots: 13
effective token ratio : 3.92x

[depth-wise] mean gate weight, top layers
     layer 1: 0.1844
   embedding: 0.1674
     layer 2: 0.0938
     layer 9: 0.0924
     layer 8: 0.0707
     layer 7: 0.0685

[width-wise] highest learned source-mass tokens
  pos=461  mass=0.05767  token='<|begin_of_text|>'
  pos=492  mass=0.05759  token='9'
  pos=478  mass=0.03770  token='Ġengineer'
  pos=470  mass=0.03369  token='ĠParis'
  pos=485  mass=0.03243  token='Ġcompany'
  pos=480  mass=0.03176  token='ave'
  pos=510  mass=0.02749  token='ĠParis'
  pos=491  mass=0.02733  token='188'
  pos=508  mass=0.02712  token='Ġlandmarks'
  pos=479  mass=0.02602  token='ĠGust'

[OT sanity check] first non-empty window
  window index        : 3
  total transported mass: 1.000000
  active source rows  : 51
  active target slots : 13

=== context ===
"The Eiffel Tower is located in Paris, France. It was designed by engineer "
"Gustave Eiffel's company and was completed in 1889 for the Exposition Universelle. "
"The tower is one of the best-known landmarks in Paris."

=== Compressed QA generation ===
Q: In which city is the Eiffel Tower located?
A: Paris


=== ComprExIT latent message ===
context tensor        : (1, 512)
raw compressed Z      : (1, 128, 2048)
decoder-space Z       : (1, 128, 2048)
valid context tokens  : 11
valid compression slots: 3
effective token ratio : 3.67x

[depth-wise] mean gate weight, top layers
     layer 1: 0.1792
   embedding: 0.1511
     layer 9: 0.1029
    layer 16: 0.0909
     layer 2: 0.0784
     layer 8: 0.0772

[width-wise] highest learned source-mass tokens
  pos=501  mass=0.22744  token='<|begin_of_text|>'
  pos=502  mass=0.15901  token='Alice'
  pos=505  mass=0.14727  token='ĠBeijing'
  pos=510  mass=0.11208  token='Ġteacher'
  pos=506  mass=0.07298  token='Ġand'
  pos=507  mass=0.06272  token='Ġworks'
  pos=503  mass=0.05758  token='Ġlives'
  pos=511  mass=0.04903  token='.'
  pos=508  mass=0.04414  token='Ġas'
  pos=509  mass=0.03449  token='Ġa'

[OT sanity check] first non-empty window
  window index        : 3
  total transported mass: 1.000000
  active source rows  : 11
  active target slots : 3

=== context ===
"Alice lives in Beijing and works as a teacher."


=== Compressed QA generation ===
Q: Which country does Alice live?
A: China


=== ComprExIT latent message ===
context tensor        : (1, 512)
raw compressed Z      : (1, 128, 2048)
decoder-space Z       : (1, 128, 2048)
valid context tokens  : 20
valid compression slots: 5
effective token ratio : 4.00x

[depth-wise] mean gate weight, top layers
     layer 1: 0.1860
   embedding: 0.1626
     layer 9: 0.1093
     layer 2: 0.0953
     layer 8: 0.0684
     layer 7: 0.0619

[width-wise] highest learned source-mass tokens
  pos=492  mass=0.14220  token='<|begin_of_text|>'
  pos=502  mass=0.10502  token='ĠLyon'
  pos=510  mass=0.08990  token='9'
  pos=507  mass=0.08208  token='ĠApril'
  pos=504  mass=0.06182  token='ĠFrance'
  pos=498  mass=0.05962  token='Ġoutdoor'
  pos=494  mass=0.04767  token='Ġteam'
  pos=497  mass=0.04229  token='Ġfirst'
  pos=500  mass=0.04132  token='Ġtrial'
  pos=499  mass=0.04009  token='Ġfield'

[OT sanity check] first non-empty window
  window index        : 3
  total transported mass: 1.000000
  active source rows  : 20
  active target slots : 5

--context "The team conducted its first outdoor field trial in Lyon, France, in April 2019."

=== Compressed QA generation ===
Q: In which city did the team conduct its first outdoor field trial?
A: Lyon, France

=== ComprExIT latent message ===
context tensor        : (1, 512)
raw compressed Z      : (1, 128, 2048)
decoder-space Z       : (1, 128, 2048)
valid context tokens  : 236
valid compression slots: 59
effective token ratio : 4.00x

[depth-wise] mean gate weight, top layers
     layer 1: 0.2003
   embedding: 0.1835
     layer 2: 0.1015
     layer 9: 0.0948
     layer 8: 0.0667
     layer 3: 0.0630

[width-wise] highest learned source-mass tokens
  pos=276  mass=0.02609  token='<|begin_of_text|>'
  pos=379  mass=0.02486  token='Ġtwelve'
  pos=363  mass=0.02328  token='ĠLyon'
  pos=467  mass=0.01980  token='ĠVienna'
  pos=325  mass=0.01974  token='ĠBerlin'
  pos=371  mass=0.01962  token='9'
  pos=304  mass=0.01779  token='ĠLondon'
  pos=441  mass=0.01772  token='ĠMadrid'
  pos=368  mass=0.01693  token='ĠApril'
  pos=380  mass=0.01608  token='Ġdevices'

[OT sanity check] first non-empty window
  window index        : 2
  total transported mass: 1.000000
  active source rows  : 108
  active target slots : 27

--context 
"In 2017, a small research team began developing a portable device for measuring river water quality. The initial planning meetings were held in London, where several engineers discussed battery life, sensor accuracy, and manufacturing costs. Early prototypes were assembled in Berlin and tested in nearby laboratories, but these tests used artificial water samples rather than water collected from rivers. After reviewing the laboratory results, the team decided to conduct its first outdoor field trial in Lyon, France, in April 2019. During that trial, researchers placed twelve devices along two sections of a river and collected measurements every thirty minutes for three weeks. The devices recorded water temperature, acidity, and dissolved oxygen levels. Heavy rain interrupted the experiment twice, and two sensors needed replacement, but most devices continued operating normally. The collected data were later analyzed by a separate group in Madrid. In November 2020, the team presented its findings at a conference in Rome. A second outdoor trial took place in Vienna in 2021, using an improved waterproof case and a larger battery. Although the project involved collaborators from many cities, the report clearly distinguished the first outdoor field trial from the earlier laboratory tests and the later conference presentation." 

=== Compressed QA generation ===
Q: In which city did the team conduct its first outdoor field trial?
A: Lyon, France

=== Compressed QA generation ===
Q: In which city were the initial planning meetings held? 
A: London

=== Compressed QA generation ===
Q: In which city were the early prototypes assembled?   
A: Berlin

=== Compressed QA generation ===
Q: In which city were the collected data analyzed? 
A: Madrid

=== Compressed QA generation ===
Q: In which city were the findings presented at a conference  
A: Rome

=== Compressed QA generation ===
Q: In which city did the second outdoor trial take place?  
A: Vienna

=== Compressed QA generation ===
Q: When did the first outdoor field trial take place?
A: April 2017  //April 2019

=== Compressed QA generation ===
Q: How many devices were deployed during the first outdoor field trial?  
A: twelve

Q: What three water properties did the devices measure?  
A: temperature, acidity, and dissolved oxygen

=== Compressed QA generation ===
Q: How long did the first outdoor field trial last?
A: three weeks

//上下文不变
=== Compressed QA generation ===
Q: In what year did the first outdoor field trial take place?
A: 2017

//更改开头年份由2017改为2016
=== Compressed QA generation ===
Q: In what year did the first outdoor field trial take place?
A: 2019

//更改context为After reviewing the laboratory results, the team decided to conduct its first outdoor field trial in Lyon, France, in April 2019
=== Compressed QA generation ===
Q: In what year did the first outdoor field trial take place?
A: April 2014

//context:The team conducted its first outdoor field trial in Lyon, France, in April 2019
=== Compressed QA generation ===
Q: In what year did the first outdoor field trial take place?
A: 2014
pos=511  mass=0.09204  token='9'

//context:The team conducted its first outdoor field trial in Lyon, France, in April 2016
=== Compressed QA generation ===
Q: In what year did the first outdoor field trial take place?
A: 2014
pos=511  mass=0.09057  token='6'

//context:The team conducted its first outdoor field trial in Lyon, France, in April 2021
=== Compressed QA generation ===
Q: In what year did the first outdoor field trial take place?
A: 2014
pos=511  mass=0.08514  token='1'

# ComprExIT 单样本问答实验整理

[原文](https://arxiv.org/abs/2602.03784)

## 1. 记录范围与实验目的

本报告依据 `expr.md` 中的原始输出整理，没有重新运行模型。目的为检查压缩后的上下文能否支持事实问答，观察有效 token 与 slot 的对应关系，并通过对照定位年份回答错误的可能来源。

原始记录共包含 19 条问答输出：基础样例 3 条、长上下文主测试 10 条、年份对照 6 条。各组用途不同，且年份对照是围绕已发现错误追加的测试，因此不合并报告为模型整体准确率。

判定采用人工语义判断，并非项目评测脚本计算的 Exact Match 或 F1。例如 `Lyon, France` 视为正确回答城市问题；`dissolved oxygen` 视为正确表达所测属性。

## 2. 实验设置与已知信息

| 项目 | 记录或说明 |
|---|---|
| 输入上下文张量 | `(1, 512)` |
| pooling 输出 | `(1, 128, 2048)` |
| projector 输出 | `(1, 128, 2048)` |
| 名义压缩率 | 4× |
| 有效压缩率 | 有效上下文 token 数 ÷ 有效 slot 数 |
| 输出形式 | 压缩诊断与单问题生成答案 |
| checkpoint | `comprexit-1b-4x-sft` |
| 模型 | `meta-llama/Llama-3.2-1B` |

所有样例中张量总长度相同，是因为输入被补齐至 512 个位置；有效内容长度不同，因此有效 slot 数不同。projector 前后形状相同不表示向量数值相同。

## 3. 基础问答与上下文长度

| 编号 | 上下文 | 有效 token | 有效 slot | 实际比例 | 问题 | 实际回答 | 判定 |
|---|---|---:|---:|---:|---|---|---|
| B1 | 埃菲尔铁塔介绍 | 51 | 13 | 3.92× | 埃菲尔铁塔在哪个城市？ | Paris | 正确 |
| B2 | Alice lives in Beijing and works as a teacher. | 11 | 3 | 3.67× | Alice 住在哪个国家？ | China | 正确，涉及背景知识 |
| B3 | 首次户外试验于 April 2019 在 Lyon 举行，句末有句号 | 20 | 5 | 4.00× | 首次户外试验在哪个城市？ | Lyon, France | 正确 |

B2 的上下文只明确给出 Beijing，未直接给出 China。回答需要结合“北京位于中国”的背景知识，因此不应把这题描述为纯粹的原文答案抽取。

长上下文包含 236 个有效 token，使用 59 个有效 slot，比例为 4.00×。与 B3 相比，两者都回答出了 Lyon，但长文本使用了更多有效 slot。

在当前窗口大小为 128、压缩率为 4、左侧 padding 的设置下，slot 以离散位置计数，末尾不足整组的有效内容仍占据一个有效 slot。因此 51/13 和 11/3 略低于 4，不代表压缩配置失效。

## 4. 同一长上下文，多问题测试

上下文描述水质设备研究项目，包含规划、组装、首次试验、数据分析、会议及第二次试验等事件。主测试共 10 题，人工语义判定为 **9/10 正确，正确率 90%**。

| 编号 | 问题要点 | 预期答案 | 实际回答 | 判定 |
|---|---|---|---|---|
| L1 | 首次户外试验城市 | Lyon, France | Lyon, France | 正确 |
| L2 | 初期规划会议城市 | London | London | 正确 |
| L3 | 早期原型组装城市 | Berlin | Berlin | 正确 |
| L4 | 数据分析城市 | Madrid | Madrid | 正确 |
| L5 | 成果发布会议城市 | Rome | Rome | 正确 |
| L6 | 第二次户外试验城市 | Vienna | Vienna | 正确 |
| L7 | 首次户外试验时间 | April 2019 | April 2017 | 错误：月份正确，年份错误 |
| L8 | 首次试验设备数量 | twelve | twelve | 正确 |
| L9 | 三种水质测量属性 | temperature, acidity, dissolved oxygen levels | temperature, acidity, and dissolved oxygen | 正确 |
| L10 | 首次户外试验持续时间 | three weeks | three weeks | 正确 |

### 观察

- 六道城市问题全部正确，说明这段上下文压缩后仍可支持不同事件与城市之间的区分。
- 设备数量、测量属性及持续时间回答正确，故当前证据不支持“所有数字信息都会丢失”的说法。
- 时间题中的错误年份 2017 出现在原文的项目启动事件中，与事件时间错配的解释一致，但单个输出不足以证明具体原因。
- 90% 仅描述这段人工构造上下文上的十道题，不代表论文数据集或一般场景的准确率。

## 5. 压缩诊断的解释

### 5.1 深度方向

四组完整诊断中，layer 1 和 embedding 都是平均门控权重最高的两项。长上下文的数值分别为 0.2003 和 0.1835。

这反映了这些输入中层融合对浅层表示的相对偏重。权重是对有效 token 取平均的结果，不能解读为某层对最终答案准确率的贡献百分比；输出也只展示排名最高的六项。

同上下文仅替换问题时，压缩模块并未以问题为输入，因此相同的压缩诊断符合当前实现。原文件未逐题记录完整诊断，不能仅凭该文件逐项验证每题权重完全一致。

### 5.2 宽度方向

长上下文的高源质量 token 包含 Lyon、Vienna、Berlin、London、Madrid、twelve 等；特殊起始 token 也位于前列。

源质量衡量 OT 源端的质量分配，不是答案正确概率，也不是针对当前问题的严格归因。各窗口分别归一化，不能直接把不同长度文本或不同窗口的 mass 大小理解为绝对重要性的变化。

### 5.3 OT 窗口

长上下文左侧 padding 数为 `512 - 236 = 276`。因此：

| 窗口（从 0 编号） | 位置范围 | 有效 token | 有效 slot |
|---|---|---:|---:|
| 0 | 0–127 | 0 | 0 |
| 1 | 128–255 | 0 | 0 |
| 2 | 256–383 | 108 | 27 |
| 3 | 384–511 | 128 | 32 |
| 合计 | — | 236 | 59 |

这与输出中的 `window index: 2`、`active source rows: 108`、`active target slots: 27` 一致。诊断只展示第一个非空窗口，不是整个上下文的 OT 汇总。

`total transported mass: 1.000000` 表示该窗口传输质量归一化为 1，不代表无损保留了全部信息，也不能单独证明 Sinkhorn 的边缘约束误差已经足够小。

## 6. 年份错误的对照实验

以下六题均问：`In what year did the first outdoor field trial take place?`

| 编号 | 上下文设置 | 预期年份 | 实际回答 | 判定 |
|---|---|---|---|---|
| Y1 | 原长文本不变，只将问题改为询问年份 | 2019 | 2017 | 错误 |
| Y2 | 长文本开头的项目启动年份由 2017 改成 2016，其余不变 | 2019 | 2019 | 正确 |
| Y3 | 仅保留以 After reviewing 开头、含 April 2019 的句子，无句末句号 | 2019 | April 2014 | 错误 |
| Y4 | The team conducted ... in April 2019，无句末句号 | 2019 | 2014 | 错误 |
| Y5 | 与 Y4 相同，仅年份为 2016 | 2016 | 2014 | 错误 |
| Y6 | 与 Y4 相同，仅年份为 2021 | 2021 | 2014 | 错误 |

这一针对错误追加的诊断组为 1/6 正确。其样本选择有针对性，不应用作整体能力估计。

### 证据支持的判断

1. **只改问题措辞未消除错误。** 长文本中，无论问完整时间还是只问年份，都回答了 2017。
2. **不支持简单复制开头年份的解释。** 把开头改成 2016 后，回答是正确的 2019，而非 2016。
3. **减少上下文不保证改善。** 只保留相关句子后，仍回答了原文没有的 2014。
4. **固定句式中的年份替换未反映到答案中。** Y4–Y6 输入年份不同，输出都为 2014。
5. **数字源质量变化不保证数字问答正确。** 末尾 token 分别为 `9`、`6`、`1`，mass 分别为 0.09204、0.09057、0.08514，但答案未随之变化。

### 尚不能得出的结论

- 不能仅凭末尾 token 判断完整年份被截断；需检查完整分词和解码结果。
- 不能据此认定错误必然发生在 OT，而不是层融合、projector 或解码器使用压缩表示的环节。
- 不能认定模型在所有输入上都偏好 2014；这里只观察到几个相关短句。
- 不同问法、句式、长度和标点的实验并非都属于严格单变量对照。尤其 B3 带句号且问城市，Y4 不带句号且问年份，两者不能用于单独归因标点影响。

