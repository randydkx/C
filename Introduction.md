# ispeak项目说明
## 1.整体架构
```
.
├── Podfile 用于依赖库的添加，使用cocoapods对项目依赖进行管理，类似于maven/gradle
├── Podfile.lock
├── Pods 相关依赖库
│   ├── Alamofire 网络访问依赖，由于没有服务器，所以没有用到
│   ├── CameraBackground 相机访问接口，用户设置任意窗口相机
│   ├── Charts 折线图、表等接口，由于时间原因，未用到
│   ├── FullscreenPopGesture 划动返回等，更加丝滑
│   ├── Headers 
│   ├── KYWaterWaveView 正弦水波纹
│   ├── Local Podspecs
│   ├── MOBFoundation 语音识别sdk依赖
│   ├── Manifest.lock
│   ├── MultiToggleButton
│   ├── Pods.xcodeproj
│   ├── PopupKit 弹出视图，这个很好用，虽然项目初始的有很多弹出视图都不是基于这个的
│   ├── ProgressHUD HUD弹出视图，实际开发中比较常见，虽然比较清亮
│   ├── SnapKit 用于对元素进行布局和约束的
│   ├── SweeterSwift
│   ├── Target Support Files
│   ├── UUIDShortener 生成随机码，用于图像和音频等的数据持久化
│   └── mob_smssdk 
├── version1
│   ├── AppDelegate.swift 用于设置app初始化的各种操作，包括数据持久化和一些依赖库的初始化
│   ├── Assets.xcassets 所有需要的图片资源等都在这里
│   ├── Base.lproj
|   ├── Main.storyboard 用于前端编辑的，可以和一个controller绑定，也可以做方法/组件的绑定
│   ├── Header.h swift和oc转换的头文件
│   ├── Info.plist 一些访问控制的参数
│   ├── SceneDelegate.swift 进入app之后的
│   ├── controller 所有的控制器
│   ├── iflyMSC.framework 可达讯飞sdk，未用
│   ├── model 所有用到的类模型都在这里，基于Model+view+controller模型
│   ├── util 依赖库
│   ├── videos 几个视频的目录
│   └── view 所有view视图都在这里面，但设计的视图其实在storyboard中
├── version1.xcodeproj   下面这些是自带的，不要管
│   ├── project.pbxproj
│   ├── project.xcworkspace
│   ├── xcshareddata
│   └── xcuserdata
├── version1.xcworkspace
│   ├── contents.xcworkspacedata
│   ├── xcshareddata
│   └── xcuserdata
├── version1Tests
│   ├── Info.plist
│   └── version1Tests.swift
└── version1UITests
    ├── Info.plist
    └── version1UITests.swift

36 directories, 15 files
```
## 2.version1文件夹子目录
```
.
├── cell
│   └── Cell_One.swift 用于列表视图中一个元素的设计，应该是设计好了但是时间仓促没有大改主体中的tableview
├── community 社区功能的几个界面控制
│   ├── ThoughtsToSendController.swift 发表想法的界面设计
│   ├── communityArticlePageController.swift 文章界面，这里使用的是一个静态界面
│   ├── communityController.swift 社区控制器，其中包含三个小的viewcontroller
│   ├── communityMessageController.swift 消息控制器，是community中的一个controller界面
│   ├── communityRecommendController.swift 推荐控制器,是community中的一个controller界面
│   ├── communityTalkController.swift 圆谈控制器,是community中的一个controller界面
│   └── detailPostController.swift 双击帖子内容之后进入可以评论以及查看帖子内容的界面
├── history
│   ├── AllHistoryController.swift 所有历史记录界面，数据提取做完了，没有在页面渲染
│   ├── DayHistoryController.swift 日记录界面，数据提取也做完了
│   └── WeekHistoryController.swift 周训练记录，数据提取也做完了
├── login 更登陆注册有关的，都在这里
│   ├── ConfirmController.swift 验证码发送以及验证的控制器
│   ├── LoginController.swift 登陆界面控制器
│   ├── RegisterController.swift 注册界面控制器
│   └── SettingController.swift 这个可能是一个没开发的界面，没什么用
├── message_3 
│   ├── CommentController.swift 评论界面，信息搜出来了没做展示，用了一个陆的空前端
│   ├── ContactController.swift 
│   ├── LikeController.swift 点赞界面，信息和展示都做了，虽然不是很好，只有这个界面集成了一个列表视图的框架
│   └── PresentController.swift 礼物界面，信息搜了出来，没做展示
├── mine “我的”界面
│   ├── ChangeController.swift 点击头像之后进入修改信息controller
│   ├── ChangenameController.swift 修改姓名
│   ├── ChangesexController.swift 修改性别
│   ├── ChangesignController.swift 修改签名
│   ├── MyspaceController.swift 点击头像进入我的个人空间
│   └── OtherSpaceController.swift 点击其他人的头像进入其他人的个人空间
├── tab tab控制器，用于控制“训练”，“社区”，“我的“三个controller
│   ├── TabContainerController.swift 
│   ├── TabController.swift 控制器配置文件
│   └── tabbar.json 三个控制器的图片以及名称配置文件
├── test 所有用于测试的文件
│   ├── HomeController.swift
│   ├── lwsController.swift
│   └── voiceViewController.swift
└── train 训练界面
    ├── Algorithm.swift 所有跟口吃率计算相关的算法都在这里，口吃率计算是基于两个DP+一个修正的
    ├── Dict.swift 构建推荐句子，算法是一个简单的向量标志位抽取+二范数最优，虽然好像没什么效果
    ├── RatingController.swift 评分界面
    ├── Repository.swift 推荐语料库
    ├── StartTrainingController.swift 一段非常丑陋的代码，是训练的那个页面
    ├── TrainViewController.swift 打开app之后进入的第一个界面，在storyboard中设计过了
    ├── segmentTrainController.swift 针对训练（跟相似推荐差不多）
    ├── shareTrainController.swift 分享界面，点左上角的+分享（但社区是没有刷新的）
    └── similarRecomController.swift 相似推荐

9 directories, 41 files
```

## 3.model文件夹子目录
>swift中如果不用别人开发的持久化框架，就需要设计一个实体更加方便进行持久化相关的操作（虽然没有框架非常繁琐，但因为最初就不知道框架，所以就照没有框架的方法进行持久化）
```
.
├── AttentionExtention.swift 关注
├── DataModel.xcdatamodeld 数据持久化设计的model，很多实体类跟其中的model是对应的
│   └── DataModel.xcdatamodel
│       └── contents
├── DataModelTest.swift
├── GiftRecordExtention.swift 礼物
├── PostExtension.swift 帖子
├── Remark.swift 评论
├── TrainExtention.swift 训练记录
├── UserAllCherishExtension.swift 点赞
├── UserExtention.swift 用户
├── module.swift 12个模块的静态实体类
└── moduleInfo.plist 一种用于持久化的表，这里没有用这种，用了通用的持久化

2 directories, 12 files
```