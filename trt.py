if __name__ == '__main__':
    string = "srl"


    # 自定义排序规则函数
    def custom_sort_rule(char):
        # 定义字符的自定义排序规则
        order = {'s': 2, 'r': 1, 'l': 3}
        # 返回字符的排序值
        return order.get(char, 0)


    # 使用自定义排序规则进行排序
    sorted_string = ''.join(sorted(string, key=custom_sort_rule))
    print(sorted_string)
