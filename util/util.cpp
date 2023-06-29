#include <iostream>
#include <fstream>
#include <cstdio>
#include <vector>
#include <sstream>
#include <string>
#include <cmath>
#include <map>

using namespace std;

void printBinaryInt(int value)
{
    // 计算 int 类型的位数
    int numBits = sizeof(int) * 8;

    // 从最高位开始逐位打印
    for (int i = numBits - 1; i >= 0; --i)
    {
        // 获取第 i 位的值
        int bit = (value >> i) & 1;

        // 输出当前位的值
        std::cout << bit;
    }
    std::cout << std::endl;
}

void printBinaryLong(long value)
{
    // 计算 int 类型的位数
    int numBits = sizeof(long) * 8;

    // 从最高位开始逐位打印
    for (int i = numBits - 1; i >= 0; --i)
    {
        // 获取第 i 位的值
        int bit = (value >> i) & 1;

        // 输出当前位的值
        std::cout << bit;
        if (i == 32)
            printf("\n");
    }
    std::cout << std::endl;
}

int main(int argc, char *argv[])
{
    printf("%ld\n",sizeof(long));
    printf("%ld\n",sizeof(long long));
}
