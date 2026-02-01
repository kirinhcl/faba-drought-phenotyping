# 蚕豆干旱胁迫高通量表型组学实验 — 数据说明文档

## 一、项目背景

- **项目名称**: NaPPI Faba Drought（蚕豆干旱实验）
- **数据总量**: 5.1 GB
- **手稿文件**: `NaPPI_Faba_drought_manuscript_processing_text@12062025 - Copy.docx`

本项目基于 **高通量植物表型平台（NaPPI）**，通过自动化成像系统追踪不同蚕豆品种在水分胁迫下的生长动态。

---

## 二、实验设计

### 2.1 品种材料（44个品种）

| 来源类别 | 数量 | 举例 |
|----------|------|------|
| 瑞典地方品种 | 15 | Horshult, Gubbestad, Goteryd, Solberga, Brottby, Dalabona... |
| 芬兰地方品种 | 14 | Tyrja, Rantala, Suontakainen, Korkeamaki, Pirhonen... |
| 商业品种 | 10 | Aurora, Mélodie, ILB 938, Birgit, Tiffany, Taifun, Kontu, Fuego, Fanfare |
| 其他 | 5 | 基因库材料、北欧/卡累利亚地方品种 |

完整品种列表：

| 编号 | 品种名 | 编号 | 品种名 |
|------|--------|------|--------|
| Acc-01 | Aurora/2 | Acc-23 | Suontakainen me0302 |
| Acc-02 | Mélodie/2 | Acc-24 | Korkeamaki |
| Acc-03 | ILB 938/2 | Acc-25 | Sairala me0503 |
| Acc-04 | Birgit | Acc-26 | Horshult |
| Acc-05 | Tiffany | Acc-27 | Saari |
| Acc-06 | Taifun | Acc-28 | Gubbestad |
| Acc-07 | Hedin/2 | Acc-29 | Goteryd |
| Acc-08 | Kontu | Acc-30 | Imatra me0101 |
| Acc-09 | Fuego | Acc-31 | Lemi me0702 |
| Acc-10 | Fanfare | Acc-32 | Kokkosenkyla |
| Acc-11 | Brottby | Acc-33 | AP8308100101 |
| Acc-12 | Solberga | Acc-34 | Seikanlampi |
| Acc-13 | Karra | Acc-35 | Sairalame0401sepb |
| Acc-14 | Odenslund | Acc-36 | Aunuksen kanta |
| Acc-15 | Lovanger | Acc-37 | Bondbona fran Kurt Jonsson |
| Acc-16 | Dalabona | Acc-38 | LAITIALA AP0101 |
| Acc-17 | Ekholmen | Acc-39 | Sodergarden |
| Acc-18 | Bohus delikatess | Acc-40 | Pirhonen |
| Acc-19 | Primus | Acc-41 | Asbro |
| Acc-20 | Romfartuna | Acc-42 | HAVO ME0801 |
| Acc-21 | Tyrja me0202 | Acc-43 | Sigvard |
| Acc-22 | Rantala me1001 | Acc-44 | Osel |

### 2.2 处理设计

| 因子 | 水平 | 说明 |
|------|------|------|
| 水分处理 | 2 | 对照组 WHC-80%（充足水分）、干旱组 WHC-30%（严重水分胁迫） |
| 生物学重复 | 3 | 每个品种 x 处理组合有 3 个独立植株 |
| **总植株数** | **264株** | 44 品种 x 2 处理 x 3 重复 |

> **命名差异说明**：`FabaDr_Obs.xlsx` 中标记为 WHC-70% / WHC-40%，而 `EndPoint_Raw_FW&DW.xlsx` 和图像目录中标记为 WHC-80% / WHC-30%。同一植株（如 `24_FabaDr_001`）在两个文件中的处理标签不同，但品种分配一致。可能是实验初始目标水位与最终实际水位的差异。

### 2.3 时间线

| 关键节点 | 日期 | DAS | DAG |
|----------|------|-----|-----|
| 实验起始（DAS=0） | 2024-10-04 | 0 | -7 |
| 发芽（DAG=0） | 2024-10-11 | 7 | 0 |
| 首次成像 | 2024-10-14 | 10 | 3 |
| 末次成像 | 2024-11-18 | 45 | 38 |
| 终点收获 | 2024-11-19 | 46 | 39 |
| 实验总跨度 | 2024-09-30 ~ 2024-12-08 | -4 ~ 65 | -11 ~ 58 |

- **DAS** = Days After Sowing（播种后天数），播种日 = 2024-10-04
- **DAG** = Days After Germination（发芽后天数），发芽日 = 2024-10-11

---

## 三、数据文件详细说明

### 3.1 `00-Misc/FabaDr_Obs.xlsx` — 核心观测日志（963 KB）

包含 **3个工作表**，记录了每次成像事件的元数据和质控标注。

#### Sheet 1: RGB1（8,717 条记录 x 17 列）

| 字段 | 说明 |
|------|------|
| Measuring Date / Time | 精确到秒的测量时间 |
| Experiment ID | 实验编号（124） |
| Round Order | 成像轮次（2-23，共22轮） |
| Tray ID / Plant ID | 托盘和植株编号（如 `24_FabaDr_151`） |
| Position | 平台位置（如 A1） |
| Accession Name / Num | 品种名和编号 |
| Treatment | 水分处理（WHC-70% / WHC-40%） |
| Replicate | 重复编号 |
| DAS / DAG / Weeks | 时间坐标（Week01 ~ Week06） |
| Angle | 拍摄角度（0, 120, 240） |
| Outlier | Yes / No 异常值标记 |
| Obs | 文字观测备注 |

**异常值统计**：167条 Yes（占 1.9%），类型包括：

| 异常类型 | 数量 | 说明 |
|----------|------|------|
| height declined due to side branches | 66 | 侧枝导致高度下降 |
| height data outlier bcz plant tilted | 33 | 植株倾斜 |
| height stopped after some time | 33 | 高度停止增长 |
| plant died | 33 | 植株死亡 |
| extraordinary height increase | 1 | 异常高度增加 |
| outlier due to loosen support | 1 | 支撑松脱 |

#### Sheet 2: RGB2（2,908 条记录 x 16 列）

- 无 Angle 字段（固定角度拍摄，可能是俯视图 top-view）
- 覆盖 22 个日期，55 条异常值
- 异常类型分布与 RGB1 类似

#### Sheet 3: FC（1,320 条记录 x 12 列）

- FC 可能代表 Fluorescence Camera（荧光相机）
- 覆盖 15 个日期（仅部分拍摄日），25 条异常值
- 无品种/处理字段（仅有 Plant ID 可用于关联）

---

### 3.2 `00-Misc/DateIndex.xlsx` — 日期索引表（71行）

完整的 日期 - DAS - DAG 对照表：

| 日期范围 | DAS 范围 | DAG 范围 |
|----------|----------|----------|
| 2024-09-30 ~ 2024-12-08 | -4 ~ 65 | -11 ~ 58 |

成像轮次与日期的完整对应关系：

| 轮次 (Round) | 日期 | DAS | DAG | 所属周 |
|:---:|------|:---:|:---:|:---:|
| 2 | 2024-10-14/15 | 10-11 | 3-4 | Week01 |
| 3 | 2024-10-16 | 12 | 5 | Week01 |
| 4 | 2024-10-17 | 13 | 6 | Week01 |
| 5 | 2024-10-18 | 14 | 7 | Week01 |
| 6 | 2024-10-21 | 17 | 10 | Week02 |
| 7 | 2024-10-23 | 19 | 12 | Week02 |
| 8 | 2024-10-24 | 20 | 13 | Week02 |
| 9 | 2024-10-25 | 21 | 14 | Week02 |
| 10 | 2024-10-28 | 24 | 17 | Week03 |
| 11 | 2024-10-30 | 26 | 19 | Week03 |
| 12 | 2024-10-31 | 27 | 20 | Week03 |
| 13 | 2024-11-01 | 28 | 21 | Week03 |
| 14 | 2024-11-04 | 31 | 24 | Week04 |
| 15 | 2024-11-07 | 34 | 27 | Week04 |
| 16 | 2024-11-08 | 35 | 28 | Week04 |
| 17 | 2024-11-09 | 36 | 29 | Week04 |
| 18 | 2024-11-11 | 38 | 31 | Week05 |
| 19 | 2024-11-13 | 40 | 33 | Week05 |
| 20 | 2024-11-14 | 41 | 34 | Week05 |
| 21 | 2024-11-15 | 42 | 35 | Week05 |
| 22 | 2024-11-18 | 45 | 38 | Week06 |
| 23 | 2024-11-18 | 45 | 38 | Week06 |

---

### 3.3 `00-Misc/EndPoint_Raw_FW&DW.xlsx` — 终点生物量数据（264行 x 15列）

在 **DAS=46, DAG=39, Week06** 收获的破坏性取样数据。

#### 字段说明

| 字段 | 说明 |
|------|------|
| DAS / DAG / Weeks | 时间坐标 |
| TrayID / Tray | 托盘编号 |
| Accession Num / Name | 品种编号和名称 |
| Treatment | 水分处理（WHC-80% / WHC-30%） |
| Rep | 重复编号 |
| PlanName | 完整描述名（如 `T001 - Acc_01 - 80% - Rep1`） |
| Bag Weight (g) | 信封/袋子空重 |
| Plant+Bag FW (g) | 植物 + 袋 鲜重 |
| P+B DW (g) | 植物 + 袋 干重 |
| **FW** | **净鲜重** = Plant+Bag FW - Bag Weight |
| **DW** | **净干重** = P+B DW - Bag Weight |

#### 处理间差异统计

| 处理 | N | FW 均值 (g) | FW 范围 (g) | DW 均值 (g) | DW 范围 (g) |
|------|---|------------|-------------|------------|-------------|
| WHC-80%（对照） | 131 | **166.0** | 57.4 ~ 285.1 | **16.7** | 5.3 ~ 27.3 |
| WHC-30%（干旱） | 131 | **75.8** | 25.8 ~ 224.3 | **7.7** | 2.1 ~ 22.6 |

> 干旱处理使鲜重降低约 **54%**，干重降低约 **54%**。

---

### 3.4 `SinglePoint Datasets/Drought_Impact(DAG).xlsx` — 干旱影响时间排名（44行 x 4列）

记录每个品种在哪一天（DAG）首次出现可观测的干旱影响。

#### 字段说明

| 字段 | 说明 |
|------|------|
| Accession Name | 品种名称 |
| Drought Impact (DAG) | 首次观测到干旱影响的 DAG |
| Rank | 排名 |
| Stress Impact | 分类标签（Early / Mid / Late） |

#### 品种干旱响应分类

| 胁迫响应类别 | DAG 范围 | 品种数 | 代表品种 |
|-------------|---------|--------|---------|
| **Early（早期敏感）** | 10-20 | 14 | Pirhonen(10), ILB 938(13), Kontu(13), Tiffany(14) |
| **Mid（中期响应）** | 24-29 | 16 | Aurora(28), Mélodie(28), Fuego(28), Fanfare(28) |
| **Late（晚期耐受）** | 33-38 | 14 | Birgit(38), Primus(38), Lovanger(38), Hedin(35) |

完整排名：

| 品种 | DAG | 排名 | 分类 |
|------|:---:|:---:|------|
| Pirhonen | 10 | 1 | Early |
| ILB 938/2 | 13 | 2 | Early |
| Asbro | 13 | 2 | Early |
| Kontu | 13 | 2 | Early |
| Tiffany | 14 | 3 | Early |
| LAITIALA AP0101 | 17 | 4 | Early |
| Goteryd | 19 | 5 | Early |
| Gubbestad | 19 | 5 | Early |
| Horshult | 19 | 5 | Early |
| Bondbona fran Kurt Jonsson | 20 | 6 | Early |
| Odenslund | 20 | 6 | Early |
| Saari | 20 | 6 | Early |
| Solberga | 20 | 6 | Early |
| Taifun | 20 | 6 | Early |
| Korkeamaki | 24 | 7 | Mid |
| Aunuksen kanta | 28 | 8 | Mid |
| Aurora/2 | 28 | 8 | Mid |
| Bohus delikatess | 28 | 8 | Mid |
| Fanfare | 28 | 8 | Mid |
| Fuego | 28 | 8 | Mid |
| Kokkosenkyla | 28 | 8 | Mid |
| Mélodie/2 | 28 | 8 | Mid |
| Osel | 28 | 8 | Mid |
| Rantala me1001 | 28 | 8 | Mid |
| Sairalame0401sepb | 28 | 8 | Mid |
| Seikanlampi | 28 | 8 | Mid |
| Sodergarden | 28 | 8 | Mid |
| Ekholmen | 29 | 9 | Mid |
| Lemi me0702 | 29 | 9 | Mid |
| Brottby | 33 | 3 | Late |
| Dalabona | 33 | 3 | Late |
| HAVO ME0801 | 33 | 3 | Late |
| Imatra me0101 | 33 | 3 | Late |
| Karra | 33 | 3 | Late |
| Sairala me0503 | 34 | 4 | Late |
| Sigvard | 34 | 4 | Late |
| Suontakainen me0302 | 34 | 4 | Late |
| Tyrja me0202 | 34 | 4 | Late |
| Hedin/2 | 35 | 5 | Late |
| AP8308100101 | 38 | 6 | Late |
| Birgit | 38 | 6 | Late |
| Lovanger | 38 | 6 | Late |
| Primus | 38 | 6 | Late |
| Romfartuna | 38 | 6 | Late |

---

### 3.5 `SinglePoint Datasets/Flower_LeavesCounts.xlsx` — 花叶计数数据（264行 x 8列）

在 **2024-11-12（DAS=39, DAG=32）** 进行的单次人工计数。

#### 字段说明

| 字段 | 说明 |
|------|------|
| Measuring Date | 测量日期（2024-11-12） |
| Tray | 托盘编号 |
| Accession Name / Num | 品种名和编号 |
| Treatment | 水分处理 |
| Replicate | 重复 |
| Leaf count next to flower | 花朵旁叶片数（取值：0, 2, 3, 4, 5） |
| Flower Pres | 花朵状态 |

#### 花朵状态统计

| 处理 | With F.（有花） | No f.（无花） | Dead（死亡） |
|------|:---:|:---:|:---:|
| WHC-80%（N=132） | 104 (78.8%) | 27 (20.5%) | 1 (0.8%) |
| WHC-30%（N=132） | 98 (74.2%) | 34 (25.8%) | 0 (0.0%) |

---

### 3.6 `TimeCourse Datasets/` — 时间序列数据集（11个文件）

此目录包含从图像中自动提取或手动标注的时间序列特征数据。

#### 3.6.1 形态学数据（Morphology）

##### `RGB1_SideView_FabaDr_Manual_Morpho.xlsx` — 侧视图形态学特征（8,718行 x 26列）

从 RGB1 侧视图像中提取的植物形态学参数，每行对应一张图像。

| 字段 | 说明 | 单位 |
|------|------|------|
| AREA_CM | 植株投影面积 | cm^2 |
| PERIMETER_CM | 植株轮廓周长 | cm |
| COMPACTNESS | 紧凑度（面积与周长的比值） | 无量纲 |
| WIDTH_CM | 植株宽度 | cm |
| HEIGHT_CM | 植株高度 | cm |
| Perc_[53;63;44] | 深绿色像素百分比 | % |
| Perc_[76;91;67] | 浅绿色像素百分比 | % |

##### `RGB2_TopView_FabaDr_Manual_Morpho.xlsx` — 俯视图形态学特征（2,909行 x 29列）

从 RGB2 俯视图像中提取的冠层形态学参数，每行对应一张图像。

| 字段 | 说明 | 单位 |
|------|------|------|
| AREA_CM | 冠层投影面积 | cm^2 |
| PERIMETER_CM | 冠层轮廓周长 | cm |
| ROUNDNESS | 圆度 | 无量纲 |
| ROUNDNESS2 | 圆度（替代算法） | 无量纲 |
| ISOTROPY | 各向同性度 | 无量纲 |
| COMPACTNESS | 紧凑度 | 无量纲 |
| ECCENTRICITY | 离心率 | 无量纲 |
| RMS | 均方根 | 无量纲 |
| SOL | 实心度（Solidity） | 无量纲 |
| Perc_[53;63;44] | 深绿色像素百分比 | % |
| Perc_[76;91;67] | 浅绿色像素百分比 | % |

#### 3.6.2 植被指数数据（Vegetation Indices）

##### `RGB-Vegetative_Indices.xlsx` — 植被指数定义表（12行 x 5列）

定义了本项目使用的 **11个 RGB 植被指数**的名称、公式和用途：

| 指数 | 全称 | 公式 | 用途 |
|------|------|------|------|
| ExG | Excess Green Index | 2*G - R - B | 植被分割、冠层覆盖 |
| GREENESS | Greenness (red-weighted norm) | (2*G - R - B) / (2*R + G + B) | 跨图像绿度对比 |
| GLI | Green Leaf Index | (2*G - R - B) / (2*G + R + B) | 绿叶检测、时序绿度监测 |
| GREEN_STRENGHT | Green Strength Index | G / (R + G + B) | 相对绿度、色彩归一化 |
| NGRVI | Normalized Green-Red VI | (G - R) / (G + R) | 植被活力代理 |
| VARI | Visible Atmospherically Resistant Index | (G - R) / (G + R - B) | 鲁棒绿度指标 |
| BG_RATIO | Blue-Green Difference | B - G | 蓝色背景分离 |
| CHROMA_BASE | Chroma Base | (R + B) / G | 衰老/胁迫筛选 |
| CHROMA_RATIO | Chroma Ratio | 0.5 * G / (R + B) | 绿度主导度量 |
| CHROMA_DIFFERENCE | Chroma Difference | 0.5*(R + B) - G | 非绿色检测 |
| TGI | Triangular Greenness Index | 0.5 * (190*(R-G) - 120*(R-B)) | 叶绿素代理指标 |

##### `VegIndex_FabaDr_RGB2.xlsx` — RGB2 俯视图植被指数时序（2,909行 x 29列）

基于 RGB2 俯视图像计算的 11 个植被指数值，包含完整的实验元数据和所有指数列。

#### 3.6.3 数字生物量与植株高度

##### `DigitalBiomass_FabaDr_Auto.xlsx` — 数字生物量（2,912行 x 15列）

通过侧视图面积和俯视图面积综合计算的数字生物量估计值。

| 字段 | 说明 |
|------|------|
| RGB1_AREA_CM_Mean | 侧视图 3 个角度的平均面积 (cm^2) |
| RGB1_AREA_CM_Sum | 侧视图 3 个角度的总面积 (cm^2) |
| RGB2_AREA_CM | 俯视图冠层面积 (cm^2) |
| Digital Biomass (e+2) | 数字生物量 (x10^2) |
| Total Projected Area (e+1) | 总投影面积 (x10^1) |

##### `DigitalBiomass_Norm_FabaDr_Auto.xlsx` — 归一化数字生物量（2,855行 x 13列）

以对照组（WHC-80%）为基准的归一化数字生物量。

| 字段 | 说明 |
|------|------|
| Digital Biomass (e+2) | 原始数字生物量 |
| Digital Biomass WHC-80% (e+2) | 对照组数字生物量基线 |
| Digital Biomass Norm (e+2) | 归一化值 = (干旱 - 对照) / 对照 |

##### `PlantHeight_Norm_FabaDr_Auto.xlsx` — 归一化植株高度（8,551行 x 20列）

以对照组为基准的归一化植株高度时序数据。

| 字段 | 说明 |
|------|------|
| HEIGHT_CM | 原始高度 (cm) |
| HEIGHT_CM WHC-80% | 对照组高度基线 (cm) |
| Plant Height Norm | 归一化高度 |

#### 3.6.4 叶绿素荧光数据（Fluorescence）

##### `FCQ_FabaDr_Auto.xlsx` — 荧光参数宽格式（1,321行 x 111列）

来自荧光相机（FC）的叶绿素荧光淬灭分析数据，包含 **93个荧光参数**。

关键参数类别：

| 参数类别 | 示例字段 | 说明 |
|----------|----------|------|
| 基础荧光 | Fo, Fm, Fv, Fp | 最小荧光、最大荧光、可变荧光 |
| 最大量子产率 | QY_max | Fv/Fm，反映 PSII 最大光化学效率 |
| 光适应参数 | Fm_L1~L4, Ft_L1~L4 | 不同光强下的荧光响应 |
| 暗恢复参数 | Fm_D1~D3, Ft_D1~D3 | 暗处理后的恢复动力学 |
| 有效量子产率 | QY_L1~Lss | 实际光化学效率 |
| 非光化学淬灭 | NPQ_L1~Lss | 热耗散能力 |
| 光化学淬灭 | qP_L1~Lss, qL_L1~Lss | 反应中心开放比例 |
| 荧光下降比 | Rfd_L1~Lss | 植物活力指标 |

`_L1~L4` = 递增光强阶段，`_Lss` = 稳态光，`_D1~D3` = 暗恢复阶段

##### `FCQ_FabaDr_Auto_Reshape.xlsx` — 荧光参数长格式（13,201行 x 33列）

同一数据的长格式（reshaped）版本，`SatPulse` 列标识了测量阶段：
- `Background`: 背景暗适应
- `L1~L4`: 递增光强阶段
- `Lss`: 稳态光
- `D1~D3`: 暗恢复阶段

#### 3.6.5 环境与灌溉数据

##### `EnvData_FabaDr.xlsx` — 环境监测数据（50,799行 x 6列）

每分钟记录一次的温室环境数据。

| 字段 | 说明 | 单位 |
|------|------|------|
| Measuring Time | 精确到秒的测量时间 | datetime |
| li1_Buffer_uE | 缓冲区光照强度 | umol/m^2/s |
| t1_Buffer_C | 缓冲区温度 | C |
| rh1_Buffer_% | 缓冲区相对湿度 | % |
| t2_Tunnel_C | 通道温度 | C |
| rh2_Tunnel_% | 通道相对湿度 | % |

##### `SC_Watering_24_FabaDr_Auto.xlsx` — 自动灌溉记录（9,507行 x 24列）

每株植物每次灌溉事件的详细记录。

| 字段 | 说明 | 单位 |
|------|------|------|
| Action Time | 灌溉时段（02-Afternoon等） | - |
| Dispense | 操作类型（Water） | - |
| Weight | 灌溉前重量 | g |
| Weight After Watering | 灌溉后重量 | g |
| Water Added | 本次添加水量 | g |
| WHC.Bf | 灌溉前水分含量 | % WHC |
| WHC.Af | 灌溉后水分含量 | % WHC |
| Time Interval | 距上次灌溉时间间隔 | hours |
| Water Loss | 两次灌溉间的水分损失 | g |
| Water Loss per Hours | 每小时水分损失率 | g/h |

---

### 3.7 `EndPoint Datasets/` — 终点汇总数据集（3个文件）

#### `BiologicalWUE_FW_DW_FabaDr_Auto.xlsx` — 生物学水分利用效率（265行 x 10列）

基于实际收获的鲜重/干重和累计灌溉量计算的水分利用效率。

| 字段 | 说明 |
|------|------|
| Fresh Weight / Dry Weight | 终点鲜重/干重 (g) |
| Total Water Added 39DAG (kg) | 截至 39DAG 的累计灌溉量 (kg) |
| Water Use Efficiency (FW) | 鲜重 WUE = FW / Total Water (g/kg) |
| Water Use Efficiency (DW) | 干重 WUE = DW / Total Water (g/kg) |

#### `DigitalWUE_FabaDr_38DAG.xlsx` — 数字水分利用效率（269行 x 8列）

基于数字生物量（非破坏性测量）和灌溉量计算的 WUE。

| 字段 | 说明 |
|------|------|
| Digital Biomass (e+2) | 38DAG 时的数字生物量 |
| Total Water Added 37DAG (kg) | 截至 37DAG 的累计灌溉量 (kg) |
| Digital WUE (38DAG) | 数字 WUE = Digital Biomass / Water |

#### `EndPoint_CorrelationData-WithoutOutliers.xlsx` — 终点多性状关联数据（293行 x 29列）

去除异常值后的终点综合数据集，整合了所有测量维度，可直接用于相关性分析。

包含的性状维度：

| 类别 | 字段 |
|------|------|
| 侧视图形态 | SideView Plant Area, Perimeter, Width, Height |
| 俯视图形态 | TopView Canopy Area, Perimeter, ROUNDNESS, ROUNDNESS2, ISOTROPY, COMPACTNESS, ECCENTRICITY, RMS, SOL |
| 荧光参数 | Quantum Yield Max, Effective Quantum Yield, Non-Photochemical Quenching |
| 数字生物量 | Digital Biomass |
| 生物学指标 | Fresh Weight, Dry Weight, Biological WUE (FM/DW) |
| 水分效率 | Total Water Added, Digital WUE |

---

### 3.8 `Avg_and_Ranks/` — 品种均值与排名数据（12个文件）

此目录包含所有关键性状按品种 x 处理取平均后的均值及其排名，用于品种间比较。

#### 时序排名文件

| 文件名 | 行数 | 排名性状 |
|--------|------|----------|
| `DigBio_FabaDr_Auto_Avg_and_Ranks.xlsx` | 1,890 | RGB1面积、RGB2面积、数字生物量（按DAG） |
| `DigBio_Norm_FabaDr_Avg_and_Ranks.xlsx` | 932 | 归一化数字生物量（按DAG） |
| `PlantHeight_Norm_FabaDr_Avg_and_Ranks.xlsx` | 932 | 归一化植株高度（按DAG） |
| `RGB1_FabaDr_Manual_Morpho_Avg_and_Ranks.xlsx` | 1,890 | 高度和宽度（按DAG） |
| `df_FabaDr_Morpho_Avg_and_Ranks.xlsx` | 1,890 | 综合形态排名：高度、宽度、RGB1面积、RGB2面积、数字生物量、归一化指标（按DAG） |
| `FCQ_FabaDr_ByDAG_Avg_and_Ranks.xlsx` | 1,296 | 荧光参数 QY_Lss, NPQ_Lss（按DAG） |
| `FCQ_FabaDr_ByWeeks_Avg_and_Ranks.xlsx` | 441 | 荧光参数 QY_Lss, NPQ_Lss（按周） |

#### 终点排名文件

| 文件名 | 行数 | 排名性状 |
|--------|------|----------|
| `EndPoint_FW&DW_Avg_and_Ranks.xlsx` | 89 | 终点 FW/DW 排名 |
| `EndPoint_FW&DW_Avg_and_Ranks_updated.xlsx` | 89 | 终点 FW/DW 排名（更新版） |
| `df_FabaDr_EndPoint-All_Avg_and_Ranks.xlsx` | 89 | 综合终点排名：38DAG 高度/宽度/面积/数字生物量/归一化指标 + 39DAG FW/DW + 荧光参数 |
| `df_FabaDr_EndPoint-Selected_RanksOnly.xlsx` | 45 | 精选排名 + Top5/Last5 计数（仅 WHC-30%） |
| `Avg_Ranks_Morpho_Diff.xlsx` | 45 | 处理间差异排名：高度差、面积差、生物量差、FW/DW差 |

#### `df_FabaDr_EndPoint-Selected_RanksOnly.xlsx` 关键字段

| 字段 | 说明 |
|------|------|
| HEIGHT_MM, 38DAG | 38DAG 高度排名 |
| WIDTH_MM, 38DAG | 38DAG 宽度排名 |
| Digital Biomass, 38DAG | 38DAG 数字生物量排名 |
| Plant Height Norm, 38DAG | 归一化高度排名 |
| Digital Biomass Norm, 38DAG | 归一化生物量排名 |
| FW, 39DAG / DW, 39DAG | 终点鲜重/干重排名 |
| QY_Lss / NPQ_Lss | 荧光参数排名 |
| **Count Top 5** | 进入前5名的次数 |
| **Count Last 5** | 进入后5名的次数 |

---

### 3.9 `00-Misc/GraphSize.txt` — 图形尺寸配置

```
200
70
1200
500
```

用于可视化脚本中设定输出图形尺寸的配置参数。

---

## 四、图像数据详细结构

### 4.1 侧视图（side_view）与俯视图（top_view）

本项目包含两种视角的图像数据：

| 属性 | side_view (RGB1) | top_view (RGB2) |
|------|------------------|-----------------|
| 总数量 | **8,718 张** | **2,908 张** |
| 分辨率 | 2560 x 3476 px | 2560 x 1920 px |
| 色彩模式 | RGBA（4通道含透明度） | RGB（3通道） |
| 拍摄角度 | 3个（0, 120, 240度） | 固定俯视 |
| 文件名标识 | `RGB1-{angle}` | `RGB2` |
| 目录结构 | `Acc/WHC/Rep/angle/` | `Acc/WHC/Rep/` |
| 处理方式 | FishEye 校正 + 黑色背景遮罩 | FishEye 校正 + 背景遮罩 |

**总图像数**: 8,718 + 2,908 = **11,626 张**

### 4.2 侧视图规格

### 4.1 图像规格

| 属性 | 值 |
|------|-----|
| 总数量 | **8,718 张** |
| 分辨率 | **2560 x 3476 像素** |
| 色彩模式 | RGBA（4通道含透明度） |
| 格式 | PNG |
| 单张大小 | ~254 KB |
| 处理方式 | FishEye 鱼眼畸变校正 + 背景遮罩（黑色背景） |
| 内容 | 植株完整侧视图，包含支撑杆，已去除盆栽和背景 |

### 4.3 成像轮次与频率

共 **22轮**（Round 2-23），按拍摄模式可分为两类：

| 类型 | 轮次编号 | 图像数/轮 | 说明 |
|------|---------|----------|------|
| 全角度扫描 (3-angle) | R2, R6, R10, R14, R18, R23 | ~792 | 264株 x 3角度，约每周一次 |
| 单角度扫描 (1-angle) | R3-5, R7-9, R11-13, R15-17, R19-21, R22 | ~264 | 264株 x 1角度，间隔日拍摄 |

全角度扫描对应 **Week 1-6** 的周初，提供完整的 3D 旋转视图。

各轮次图像数量：

| Round | 图像数 | 类型 | Round | 图像数 | 类型 |
|:---:|:---:|------|:---:|:---:|------|
| 2 | 792 | 全角度 | 13 | 264 | 单角度 |
| 3 | 264 | 单角度 | 14 | 792 | 全角度 |
| 4 | 264 | 单角度 | 15 | 264 | 单角度 |
| 5 | 264 | 单角度 | 16 | 264 | 单角度 |
| 6 | 792 | 全角度 | 17 | 264 | 单角度 |
| 7 | 264 | 单角度 | 18 | 791 | 全角度 |
| 8 | 264 | 单角度 | 19 | 264 | 单角度 |
| 9 | 264 | 单角度 | 20 | 264 | 单角度 |
| 10 | 787 | 全角度 | 21 | 264 | 单角度 |
| 11 | 264 | 单角度 | 22 | 186 | 部分 |
| 12 | 264 | 单角度 | 23 | 618 | 全角度 |

### 4.4 目录结构

```
img/
├── side_view/                           <- RGB1 侧视图
│   ├── Acc-01 - Aurora_2/
│   │   ├── WHC-80%/
│   │   │   ├── Rep-1 - 24_FabaDr_001/
│   │   │   │   ├── 000/                <- 0 度角度
│   │   │   │   │   ├── 124-2-24_FabaDr_001-RGB1-000-FishEyeMasked.png
│   │   │   │   │   ├── 124-6-24_FabaDr_001-RGB1-000-FishEyeMasked.png
│   │   │   │   │   └── ... (约11张)
│   │   │   │   ├── 120/                <- 120 度角度
│   │   │   │   └── 240/                <- 240 度角度
│   │   │   ├── Rep-2 - 24_FabaDr_002/
│   │   │   └── Rep-3 - 24_FabaDr_003/
│   │   └── WHC-30%/
│   │       ├── Rep-1 - 24_FabaDr_004/
│   │       └── ...
│   ├── Acc-02 - Melodie_2/
│   └── ... (共44个品种目录)
│
└── top_view/                            <- RGB2 俯视图
    ├── Acc-01 - Aurora_2/
    │   ├── WHC-80%/
    │   │   ├── Rep-1 - 24_FabaDr_001/   <- 无角度子目录，直接存放图片
    │   │   │   ├── 124-2-24_FabaDr_001-RGB2-FishEyeMasked.png
    │   │   │   ├── 124-6-24_FabaDr_001-RGB2-FishEyeMasked.png
    │   │   │   └── ... (约11张)
    │   │   └── ...
    │   └── WHC-30%/
    └── ... (共44个品种目录)
```

### 4.5 文件名解码

```
124-10-24_FabaDr_151-RGB1-000-FishEyeMasked.png
 |   |  |     |   |    |    |         |
 |   |  |     |   |    |    |         +-- 鱼眼校正 + 背景遮罩
 |   |  |     |   |    |    +------------ 角度: 000 / 120 / 240
 |   |  |     |   |    +----------------- 相机: RGB1 (侧视彩色相机)
 |   |  |     |   +---------------------- 植株ID: FabaDr_151
 |   |  |     +-------------------------- 项目: FabaDr (Faba Drought)
 |   |  +-------------------------------- 年份后缀: 24 (2024)
 |   +----------------------------------- Round Order: 10 (成像第10轮)
 +--------------------------------------- 实验ID: 124
```

### 4.6 数据完整性

| 指标 | 值 |
|------|-----|
| 标准角度文件夹总数 | 792 (264 植株 x 3 角度) |
| 包含 11 张图的文件夹（标准） | 774 (97.7%) |
| 包含 12 张图的文件夹（多1张） | 12 (Acc-02 Rep-2, Acc-22 Rep-2, Acc-38 Rep-2) |
| 包含 10 张图的文件夹（少1张） | 6 (Acc-34, Acc-36, Acc-44 部分) |

> 每个角度文件夹仅含约 **11张**（非22轮全部）——图像目录只保留了全角度扫描轮次的图片，单角度轮次的图像可能存储在别处或尚未纳入本数据集。

### 4.7 图像内容描述

- **对照组（WHC-80%）**: 植株挺立，叶片展开，色泽浓绿，茎秆粗壮
- **干旱组（WHC-30%）**: 植株矮小，叶片萎蔫下垂，色泽灰绿，茎秆细弱，生物量明显减少

---

## 五、数据关联关系

```
                         DateIndex.xlsx
                    (日期/DAS/DAG 对照表)
                             |
            +----------------+----------------+
            |                |                |
            v                v                v
     FabaDr_Obs.xlsx   TimeCourse数据集   EndPoint数据集
     (RGB1/RGB2/FC     +-----------+     +------------+
      成像事件日志)     | Morpho    |     | FW/DW      |
            |          | VI        |     | Bio WUE    |
            |          | DigBio    |     | Dig WUE    |
            |          | Height    |     | Correlation|
            v          | FCQ       |     +-----+------+
    img/side_view/     | Env       |           |
    img/top_view/      | Watering  |           |
    (11,626张图像)     +-----+-----+           |
            |                |                 |
            +--------+-------+---------+-------+
                     |                 |
                     v                 v
              SinglePoint数据集    Avg_and_Ranks/
              +-------------+     (品种均值与排名)
              | Drought     |     +--------------+
              |  Impact     |     | 时序排名     |
              | Flower/     |     | 终点排名     |
              |  Leaves     |     | 综合排名     |
              +-------------+     +--------------+

    关联键: Plant ID / Tray ID (如 24_FabaDr_001)
            Accession Num (如 Acc-01)
            Treatment / Replicate / DAS / DAG
```

### 关联键说明

| 关联键 | 说明 | 示例 |
|--------|------|------|
| Plant ID / Tray ID | 植株唯一标识 | `24_FabaDr_001` |
| Accession Num | 品种编号 | `Acc-01` |
| Accession Name | 品种名称 | `Aurora/2` |
| Treatment | 水分处理 | `WHC-80%` / `WHC-30%` |
| Replicate | 重复编号 | `Rep-1` |
| Round Order | 成像轮次（对应图像文件名中的数字） | `10` -> `124-10-24_...` |
| DAS / DAG | 时间坐标（可通过 DateIndex 互转） | DAS=10 <-> DAG=3 |
| Weeks | 周编号 | `Week01` ~ `Week06` |

---

## 六、完整文件清单

### 6.1 目录结构总览

```
data/
├── 00-Misc/                          <- 核心元数据和原始记录
│   ├── FabaDr_Obs.xlsx                  (963 KB, 成像事件日志)
│   ├── DateIndex.xlsx                   (14 KB, 日期索引)
│   ├── EndPoint_Raw_FW&DW.xlsx          (37 KB, 终点生物量原始数据)
│   ├── GraphSize.txt                    (图形尺寸参数)
│   └── NaPPI_Faba_drought_manuscript... (手稿文本)
│
├── TimeCourse Datasets/              <- 时间序列提取特征 (11个文件)
│   ├── RGB1_SideView_FabaDr_Manual_Morpho.xlsx     (侧视图形态学)
│   ├── RGB2_TopView_FabaDr_Manual_Morpho.xlsx      (俯视图形态学)
│   ├── RGB-Vegetative_Indices.xlsx                  (植被指数定义)
│   ├── VegIndex_FabaDr_RGB2.xlsx                    (RGB2植被指数值)
│   ├── DigitalBiomass_FabaDr_Auto.xlsx              (数字生物量)
│   ├── DigitalBiomass_Norm_FabaDr_Auto.xlsx         (归一化数字生物量)
│   ├── PlantHeight_Norm_FabaDr_Auto.xlsx            (归一化植株高度)
│   ├── FCQ_FabaDr_Auto.xlsx                         (荧光参数-宽格式)
│   ├── FCQ_FabaDr_Auto_Reshape.xlsx                 (荧光参数-长格式)
│   ├── EnvData_FabaDr.xlsx                          (环境监测数据)
│   └── SC_Watering_24_FabaDr_Auto.xlsx              (灌溉记录)
│
├── SinglePoint Datasets/             <- 单时间点数据 (2个文件)
│   ├── Drought_Impact(DAG).xlsx                     (干旱响应排名)
│   └── Flower_LeavesCounts.xlsx                     (花叶计数)
│
├── EndPoint Datasets/                <- 终点汇总数据 (3个文件)
│   ├── BiologicalWUE_FW_DW_FabaDr_Auto.xlsx        (生物学WUE)
│   ├── DigitalWUE_FabaDr_38DAG.xlsx                 (数字WUE)
│   └── EndPoint_CorrelationData-WithoutOutliers.xlsx (多性状关联数据)
│
├── Avg_and_Ranks/                    <- 品种排名数据 (12个文件)
│   ├── DigBio_FabaDr_Auto_Avg_and_Ranks.xlsx
│   ├── DigBio_Norm_FabaDr_Avg_and_Ranks.xlsx
│   ├── PlantHeight_Norm_FabaDr_Avg_and_Ranks.xlsx
│   ├── RGB1_FabaDr_Manual_Morpho_Avg_and_Ranks.xlsx
│   ├── df_FabaDr_Morpho_Avg_and_Ranks.xlsx
│   ├── FCQ_FabaDr_ByDAG_Avg_and_Ranks.xlsx
│   ├── FCQ_FabaDr_ByWeeks_Avg_and_Ranks.xlsx
│   ├── EndPoint_FW&DW_Avg_and_Ranks.xlsx
│   ├── EndPoint_FW&DW_Avg_and_Ranks_updated.xlsx
│   ├── df_FabaDr_EndPoint-All_Avg_and_Ranks.xlsx
│   ├── df_FabaDr_EndPoint-Selected_RanksOnly.xlsx
│   └── Avg_Ranks_Morpho_Diff.xlsx
│
└── img/                              <- 原始图像 (11,626张)
    ├── side_view/                       (8,718张, RGB1, 2560x3476, RGBA)
    └── top_view/                        (2,908张, RGB2, 2560x1920, RGB)
```

### 6.2 数据摘要

| 维度 | 值 |
|------|-----|
| 品种数 | 44 |
| 水分处理 | 2 (WHC-80%, WHC-30%) |
| 生物学重复 | 3 |
| 总植株数 | 264 |
| 成像轮次 | 22 (Round 2-23) |
| 实验周数 | 6 (Week01-Week06) |
| 拍摄角度 | 3 (0, 120, 240) |
| 侧视图(RGB1)图像数 | 8,718 |
| 俯视图(RGB2)图像数 | 2,908 |
| **总图像数** | **11,626** |
| 侧视图分辨率 | 2560 x 3476 px (RGBA) |
| 俯视图分辨率 | 2560 x 1920 px (RGB) |
| RGB1 形态学记录 | 8,718 |
| RGB2 形态学记录 | 2,909 |
| 植被指数种类 | 11 |
| 荧光参数数量 | 93 |
| 环境监测记录 | 50,799（约每分钟1条） |
| 灌溉记录 | 9,507 |
| 终点生物量记录 | 264 |
| 花叶计数记录 | 264 |
| WUE 记录 | 265 (生物学) + 269 (数字) |
| 品种排名文件 | 12 |
| **总Excel文件数** | **28** |
