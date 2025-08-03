# BlindStar 决策层 Context

> 本文件描述在 BlindStar 系统中整合 **YOLO 语义检测** 与 **深度·速度几何信息** 的决策框架，为未来路径规划、语音播报与 haptic 反馈模块提供统一数据语义与优先级规范。

---

## 1. 感知输入标准化

| 输入字段 | 来源 | 数据类型 | 说明 |
|----------|------|----------|------|
| `class_id`         | YOLO / OCR | int    | 语义类别 ID (COCO/自定义 0-N) |
| `class_name`       | YOLO / OCR | str    | 语义类别名称 |
| `confidence`       | YOLO       | float  | 检测置信度 0-1 |
| `bbox`             | YOLO       | list[float] | `[x1,y1,x2,y2]` 像素坐标 |
| `distance_m`       | MiDaS Depth| float? | 物体中心距摄像头 (m)；若 <0 表示未知 |
| `velocity_mps`     | Tracking   | float? | 物体相对摄像头速度 (正值=靠近) |
| `height_m`         | Depth map  | float? | 物体最高点至地面的实际高度 |
| `time_to_collide`  | 计算字段    | float? | TTC = distance / max(velocity, ε) |
| `is_static`        | Tracking   | bool   | 连续 N 帧速度 < 阈值视为静止 |

> ⚠️ **缺失字段以 `null` 传递**，上层决策须具备容错逻辑。

---

## 2. 语义分级

| Level | 颜色 | 范例类别                         | 决策约束 |
|-------|------|----------------------------------|----------|
| L0    | 🟢  | 路面标线、导盲砖、地面箭头           | 提示 or 辅助指引，不阻挡通行 |
| L1    | 🟡  | 行人、自行车、宠物、路灯杆、垃圾桶   | 避让/绕行，语音提醒“注意” |
| L2    | 🟠  | 汽车、摩托车、公交、卡车            | 高危险动体，提前播报 + 规划最短侧向距离 |
| L3    | 🔴  | 交通灯(红)、停止/让行标志、施工护栏 | 法规硬约束：禁止前进或必须停下 |

决策时按最高风险等级覆盖低级别，若同一像素区域有多对象保留 Lmax。

---

## 3. 风险评估逻辑

```
if obj.level == L3:
    action = STOP
elif obj.level == L2 and obj.time_to_collide < 3s:
    action = BRAKE_AND_WARN
elif obj.level == L1 and obj.distance_m < 1.2:
    action = SIDE_STEP
elif obj.level == L0:
    action = FOLLOW_GUIDE
else:
    action = CONTINUE
```

- `SIDE_STEP` 依赖深度地图寻找左右 45° 可行空隙 > 0.8 m。  
- 在存在多个对象时取 **最危险对象** 的建议动作。

---

## 4. 模块接口草案

### 4.1 Perception → Decision

```jsonc
{
  "frame_id": 1024,
  "timestamp": 1723688842.112,
  "camera_pose": [0,0,0],
  "objects": [ {…}, {…} ]  // 上表字段集合
}
```

### 4.2 Decision → Actuator (TTS/Haptic)

```jsonc
{
  "frame_id": 1024,
  "action": "BRAKE_AND_WARN",
  "speech": "前方两米处来车, 请停下等待",
  "vibration_pattern": "long",
  "target_bearing": -15.0      // 角度, 左负右正
}
```

---

## 5. TODO / 扩展

1. **天气感知**：融合雨雪检测调高误差容忍度。  
2. **用户状态**：结合 IMU / GPS 速度确定自身体动 vs. 静止。  
3. **学习型决策**：记录用户反馈(走/停)训练 RL 策略。  
4. **紧急通话触发**：L3 级危险持续 >5 s 触发 SOS 消息。

---

保持此 context 与代码同步，任何决策字段或风险级别调整请先更新本文件。 