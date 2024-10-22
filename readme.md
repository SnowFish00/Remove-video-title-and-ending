## 入口

begin 下的 main.go 

`运行指令 go run main.go`

## 自动去片头片尾

project 下的 list-bar.py

## 资源

视频放在 resource下的 mp4-in/A(注意A的名字随意，但是要与图片文件夹名字一样) 格式为mp4

(header-after.png)片头，片尾(tail-before.png)放在 resource 下的cut/A 格式为png(注意这个片头片尾的图片要尽可能做到，片头的最后一张、片尾的第一张)

## 注意

注意修改py中的输出绝对路径 和输入绝对路径

若匹配错误则提高 tolerance 的期待值 若匹配失败则 降低tolerance 的的期待值

允许手动微调整 时间

header_final_cut_time = header_cut_time + 0.1

tail_final_cut_time = tail_cut_time - 0.1

## 重要

该项目只作为学习交流，请勿触碰视频源公司的利益，请遵循相关法律，在此免责声明。