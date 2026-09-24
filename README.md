# 口袋毛孩 · Flutter 1.0 真机测试版（离线抠图版）

拍照 → 自动抠图 → 屏幕中央的一比一真实宠物。

- **iOS**：Vision Framework 前景分割（iOS 17+，系统内置模型）
- **Android**：TensorFlow Lite + U2-Net 轻量版（**完全离线，无 Google 服务依赖，华为无 GMS 手机可直接跑**）

## 目录

```
pocket_kitty_flutter/
├── lib/main.dart                      # 全部 Flutter 代码（选图 / 抠图调度 / 舞台动效）
├── pubspec.yaml                       # 依赖：image_picker + audioplayers
├── .github/workflows/build-apk.yml    # GitHub Actions 云端自动打包
├── ios/Runner/
│   ├── AppDelegate.swift              # Vision 前景分割实现
│   └── Info.plist                     # 相册权限文案
└── android/
    ├── app/build.gradle               # TFLite 依赖（无任何 play-services 引用）
    ├── app/src/main/assets/u2netp.tflite   # 抠图模型，已内置，无需下载
    ├── app/src/main/AndroidManifest.xml    # 无网络权限、无 Google 服务声明
    └── app/src/main/kotlin/.../MainActivity.kt  # TFLite 推理 + 遮罩合成
```

## 抠图模型（已放进工程，不用再下）

| 项 | 值 |
|---|---|
| 文件 | `android/app/src/main/assets/u2netp.tflite` |
| 来源 | https://huggingface.co/abhimanyu666/u2nettflite/resolve/main/u2netp.tflite |
| 大小 | 4.4 MB（打包进 APK，运行时零网络请求） |
| 输入 | 1×320×320×3 float32，ImageNet mean/std 归一化 |
| 许可 | Apache-2.0 |

想换更高质量的全量版：下载同仓库的 `u2net.tflite`（84 MB）放进 assets，
把 `MainActivity.kt` 里 `MODEL_FILE` 改成 `"u2net.tflite"` 即可，其余代码不用动。

## 云端打包 APK（不需要配任何本地环境）

1. 在 GitHub 新建一个仓库，把 `pocket_kitty_flutter/` 整个文件夹推上去
2. 仓库页 → **Actions** 标签 → 左侧 **Build Android APK** → **Run workflow**
3. 等 5-10 分钟，跑完后在该次运行页面底部的 **Artifacts** 下载 `pocket-kitty-apk`
4. 解压得到 `app-release.apk`

推代码后每次 push 也会自动打包。

## 华为手机安装 APK

1. 先把 `app-release.apk` 传到手机：微信「文件传输助手」/ QQ / 华为分享 / 数据线都行
2. 手机上点开 APK → 系统会提示禁止安装 → 选择**仍要安装**或去设置授权：
   - **HarmonyOS / EMUI 新版**：设置 → 安全 → 更多安全设置 → 安装外部来源应用 → 给「文件管理 / 微信 / 浏览器」打开允许
   - 如果有**纯净模式**拦截：设置 → 系统和更新 → 纯净模式 → 关闭（或设置 → 安全 → 纯净模式）
3. 返回点开 APK 正常安装即可

## 测试素材怎么放

- **三张猫猫照片**：不需要放到任何特殊目录。用微信/数据线把「芝士」的三张照片传进手机相册，
  打开 App 点「选择本地照片」，一次多选这三张即可，抠图自动完成
- **meow.mp3**：放到工程的 `pocket_kitty_flutter/assets/sounds/meow.mp3`，
  取消 `pubspec.yaml` 里 assets 两行注释，重新推送打包。
  没放音频文件也不影响运行——点击宠物会自动用系统提示音代替

## 本地有环境的话

```bash
cd pocket_kitty_flutter
flutter create . --org com.pocketkitty --project-name pocket_kitty --platforms android
flutter pub get
flutter run
```

## 已知边界（实测时留意）

- U2-Net 是通用显著性分割，对橘猫这类高对比主体效果好；若毛发边缘不够细，
  先换全量版 `u2net.tflite` 对比，还不满意再换 ISNet 模型（可以做，代码结构已预留）
- 模型有 7 个输出张量（U2-Net 的多侧输出），代码默认取第 1 个（最终融合图）；
  若实机效果边缘偏粗，把 `MainActivity.kt` 里 `buffers[0]` 的下标换成其它 0-6 试一下即可
- 首次点击抠图比后续慢（模型初始化预热），属正常现象
- iOS 分支保持不变：Vision 方案要求 iOS 17+
