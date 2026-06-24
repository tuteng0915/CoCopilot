# Math-to-Code Refiner Case Study

**Setting**: AR model (Llama-3.1-8B-Instruct) generates Python `solution()` from GSM8K word problems.
Dream-Coder refiner then re-scores each token, masks low-confidence positions, and regenerates.
Overall accuracy shift: AR 74.8% → +Dream 75.8% (+1.0pp).

---

## 总体分类统计

| 分类 | 数量 | 占 wrong cases 的比例 | 含义 |
|------|------|----------------------|------|
| Both correct（皆对） | 979 | — | AR 本身已对，refiner 无需干预 |
| **Confident Wrong**（置信度高但错）| 240 | **72%** | 所有 token 置信度 ≥ τ，refiner 不 mask，代码不变，仍错 |
| Changed + still wrong（改了仍错） | 72 | 22% | Refiner mask 了某些 token，regenerate 后还是错 |
| **Fixed**（成功修复） | **21** | **6.3%** | Refiner 正确识别出错误 token，regenerate 后对了 |
| Degraded（改坏了） | 7 | 2.1% | AR 本来对的，refiner 改错了 |

**核心结论**：72% 的错误案例属于 "Confident Wrong"——AR 用正确的 Python 语法写出了错误的推理逻辑，每个 token 都看起来合理，置信度高，refiner 没有任何 mask 的依据。

---

## Category 1: ✅ Fixed — Refiner 成功修复

这类案例的共同特征：**错误是实现层面的笔误或符号方向错误**，且代码注释或变量名提供了"正确值的上下文"，使 Dream 能够识别出不一致的 token。

---

### Case F-1：数字 copy 错误（`12` vs `11`）

**问题**：A whatchamacallit spins eleven times faster than a thingamabob. A whatchamacallit spins at 121 m/s. How fast does a whirligig spin?

| | AR | Dream |
|--|--|--|
| 关键行 | `speed_thingamabob = 121 / 12` | `speed_thingamabob = 121 / 11` |
| 预测答案 | 50.4 ❌ | **55.0 ✅** |

```python
# AR（错误）
def solution():
    speed_thingamabob = 121 / 12  # Whatchamacallit spins 11 times faster than thingamabob
    speed_whirligig = speed_thingamabob * 5
    return speed_whirligig

# Dream（修复后）
def solution():
    speed_thingamabob = 121 / 11  # Whatchamacallit spins 11 times faster than thingamabob
    speed_whirligig = speed_thingamabob * 5
    return speed_whirligig
```

**分析**：AR 将 "eleven times" 错写为除数 `12`（差 1 的笔误）。注释中明确写着 `"11 times faster"`，而代码写的是 `12`——两者不一致使得 `12` 这个 token 的置信度被 Dream 打低。Refiner 将其替换为上下文一致的 `11`，修复成功。

---

### Case F-2：运算方向错误（`+` vs `-`）

**问题**：Bob will have 60 marbles if he receives 2 dozen more. How many does he have now minus 10?

| | AR | Dream |
|--|--|--|
| 关键行 | `marbles = 60 + 2 * 12` | `marbles = 60 - 2 * 12` |
| 预测答案 | 74 ❌ | **26 ✅** |

```python
# AR（错误）
def solution():
    marbles = 60 + 2 * 12  # If he receives 2 dozen more, he will have 60
    marbles -= 10
    return marbles

# Dream（修复后）
def solution():
    marbles = 60 - 2 * 12  # If he receives 2 dozen more, he will have 60
    marbles -= 10
    return marbles
```

**分析**：逻辑上"再加 24 就有 60"意味着"现在有 60 − 24 = 36"，但 AR 写成了加法。`+` 号置信度低（与 comment 语义矛盾），Dream 将其换为 `-`。

---

### Case F-3：语义方向错误（"less" 被忽略）

**问题**：Krissa orders shirts. "Four less than the number of small students need medium."

| | AR | Dream |
|--|--|--|
| 关键行 | `medium = small + 4` | `medium = small - 4` |
| 预测答案 | 91 ❌ | **75 ✅** |

```python
# AR（错误）
def solution():
    extra_small = 11
    small = 2 * extra_small       # 22
    medium = small + 4            # 26（错：应是 small - 4 = 18）
    large = medium / 2
    extra_large = large + 6
    return extra_small + small + medium + large + extra_large

# Dream（修复后）
    medium = small - 4            # 18 ✓
```

**分析**：AR 将 "four less than" 错误地实现为 `+4`。在 small=22 的上下文中，`+4` 得到 26，而所有其他量均来自减法关系，使 `+` 的置信度偏低，Dream 将其修正。

---

### Case F-4：已有库存被错误累加（`+` vs `-`）

**问题**：John needs 40 hot dogs. He already has 4 leftover. How much to spend on new packs?

| | AR | Dream |
|--|--|--|
| 关键行 | `(guests * per_guest) + leftover_hot_dogs` | `(guests * per_guest) - leftover_hot_dogs` |
| 预测答案 | $16 ❌ | **$12 ✅** |

```python
# AR（错误）：把已有的 4 根加进了需要量，多买了一包
total_hot_dogs_needed = (total_guests * hot_dogs_per_guest) + leftover_hot_dogs  # 44
packs_needed = ceil(44 / 6)  # 8 packs → $16

# Dream（修复后）：已有库存抵扣需求
total_hot_dogs_needed = (total_guests * hot_dogs_per_guest) - leftover_hot_dogs  # 36
packs_needed = ceil(36 / 6)  # 6 packs → $12
```

**分析**：变量名 `leftover_hot_dogs` 与 `+` 操作符的语义冲突（"leftover" 表示已有，应减去），降低了该 `+` 的置信度。

---

### 修复案例共同规律

所有 21 个修复案例均属于**符号/运算符层面的一致性错误**：

| 错误类型 | 典型形式 | 出现次数 |
|----------|---------|---------|
| 数字 copy 错误 | `121/12` 而注释写 "11 times" | ~5 |
| `+` / `-` 方向反 | "receives more → 应减" 却写加 | ~10 |
| `*` / `/` 方向反 | "four less" 却写加 | ~3 |
| 其他常量偏差 | 周数 `6` 应为 `5` 等 | ~3 |

关键机制：**代码注释或变量命名提供了"预期值的语言描述"**，当生成的数值/运算符与该描述不一致时，Dream 的 MLM scorer 会对该 token 打低分。

---

## Category 2: ❌ Confident Wrong — Refiner 无能为力

这类案例的特征：**代码语法完美、逻辑自洽，但推理框架本身是错的**。所有 token 置信度均高于 τ=0.9，refiner 不做任何 mask，代码原样输出。

---

### Case C-1：累计量 vs. 终态量的混淆

**问题**：公司初始 200 名员工，每月新增 20 人，每人月薪 $4000，三个月总支出是多少？

AR 预测：$3,120,000 ❌　　正确答案：$2,880,000 ✅

```python
# AR（wrong，代码完全合法）
def solution():
    initial_employees = 200
    new_employees_per_month = 20
    salary_per_employee = 4000
    months = 3
    total_employees = initial_employees + (new_employees_per_month * months)  # 260（第 3 月末人数）
    total_salary = total_employees * salary_per_employee * months             # 用终态人数乘以 3 个月 → 多算了
    return total_salary
```

**正确逻辑**：应按月累加（月 1: 220 人，月 2: 240 人，月 3: 260 人，合计 = 720 人月 × $4000 = $2,880,000）。AR 用最终人数乘以月数，是一个**聚合逻辑错误**，代码完全合法，token 级别无任何可疑之处。Dream 输出与 AR 完全相同，结果相同。

---

### Case C-2：自然语言歧义的错误消解

**问题**：Tara bought 8 packs of 5 canvas bags for $4 each. (sold at $8 each. Profit?)

AR 预测：$288 ❌　　正确答案：$160 ✅

```python
# AR（wrong，代码完全合法）
def solution():
    cost_per_pack = 4          # AR 将 "$4 each" 理解为每包 $4
    num_packs = 8
    num_bags_per_pack = 5
    total_cost = cost_per_pack * num_packs           # 32
    total_revenue = 8 * num_packs * num_bags_per_pack  # 320
    profit = total_revenue - total_cost              # 288（错）
    return profit

# 正确理解："$4 each" = 每个 canvas bag $4
# total_cost = 4 * 8 * 5 = 160，profit = 320 - 160 = 160
```

**分析**："$4 each" 中的 "each" 指代 pack 还是 bag 是模糊的。AR 选择了错误的理解，但`cost_per_pack = 4`是完全合理的 Python 赋值，无任何语言信号让 Dream 质疑它。Refiner 无法知道 "each" 所指为何。

---

## Category 3: 🔄 Changed But Still Wrong — 改了也没用

这类案例：refiner 找到了低置信 token 并修改，但修改后的代码仍然错误。根本原因往往是**错误存在于 refiner 的盲区**（常量初始化错误，而非运算符方向）。

---

### Case W-1：根因在初始化，而非运算符

**问题**：Kelly 有 5 packs of bacon（共 $10），6 packets of chicken（每包是 bacon 每包的 2 倍），预算 $65，剩多少？

正确答案：$5 ✅（bacon 每包 $2，chicken 每包 $4，总支出 10+24+12+14=60，剩 5）

```python
# AR（错误 1：bacon_price=10 视为每包价，实为总价）
def solution():
    bacon_price = 10            # ← 根本错误：应为 10/5=2
    chicken_price = bacon_price * 2   # 20（每包），总计 120 → 远超预算
    ...
    return budget - total_cost  # -131

# Dream（错误 2：改了乘法为除法，但根因未修）
def solution():
    bacon_price = 10            # ← 根本错误仍在
    chicken_price = bacon_price / 2   # 5（每包），总计 30 → 仍然错
    ...
    return budget - total_cost  # -41
```

**分析**：Dream 注意到 `chicken_price = bacon_price * 2` 中 `* 2` 置信度低（因为结果 20 与 chicken_price 在后续计算中产生异常大的总价），于是改为 `/ 2`。但根本错误在第一行 `bacon_price = 10`——这是对"总价"和"单价"的混淆。`10` 这个数字是从题目直接抄来的，置信度很高，refiner 不会质疑它。

---

## 总结：Refiner 的能力边界

```
  GSM8K wrong cases (333 total)
  │
  ├─ 72%: Confident Wrong ──────────── Refiner 看不到问题（无低置信 token）
  │        推理框架错 / 歧义理解错         代码原样输出
  │
  ├─ 22%: Changed, Still Wrong ──────── Refiner 改了，但改错了地方
  │        根本错误在高置信区域             局部修改无法修复全局推理
  │
  └─  6%: Fixed ──────────────────────── Refiner 成功
           低置信 token = 真正的笔误        符号/运算符不一致性可被检测
           （注释 vs 代码的语义矛盾）
```

**Refiner 有效的充分条件**：
1. 错误 token 本身置信度低（即 AR 在写这个 token 时"自己也不确定"）
2. 周围的注释/变量名提供了可参照的正确语义

**Refiner 无效的根本原因**：
- 大多数推理错误的代码，AR **写得很自信**（confident wrong）
- 置信度是**语法/代码层面**的置信度，无法感知**数学语义**层面的正确性
- Refiner 是代码实现编辑器，不是数学推理引擎

这一分析精确划定了 CoCoder 方法的适用范围：适用于代码实现层面的笔误修复（HumanEval / MBPP），不适用于嵌入了错误数学推理的代码（Math-to-Code）。
