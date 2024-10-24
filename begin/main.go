package main

import (
	"fmt"
	"os/exec"
)

func main() {
	// 定义要执行的命令
	cmd := exec.Command("python3", "../project/list-bar-v5.py")

	// 获取命令的标准输出和标准错误输出
	output, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Printf("Error executing command: %s\n", err)
		return
	}

	// 输出命令的结果
	fmt.Println(string(output))
}
