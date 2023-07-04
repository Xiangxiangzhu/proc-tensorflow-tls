import pandas as pd

# 第一个字典
dict1 = {12: '-gneE10_0', 13: '-gneE10_0', 14: '-gneE10_1', 15: '-gneE10_2',
         0: '-gneE8_0', 1: '-gneE8_0', 2: '-gneE8_1', 3: '-gneE8_2',
         8: 'gneE12_0', 9: 'gneE12_0', 10: 'gneE12_1', 11: 'gneE12_2',
         4: 'gneE7_0', 5: 'gneE7_0', 6: 'gneE7_1', 7: 'gneE7_2'}

# 第二个字典
dict2 = {12: 'r', 13: 's', 14: 's', 15: 'l',
         0: 'r', 1: 's', 2: 's', 3: 'l',
         8: 'r', 9: 's', 10: 's', 11: 'l',
         4: 'r', 5: 's', 6: 's', 7: 'l'}

# 转换为DataFrame
df1 = pd.DataFrame(list(dict1.items()), columns=['Key', 'Value1'])
df2 = pd.DataFrame(list(dict2.items()), columns=['Key', 'Value2'])

# 合并DataFrame
df_merged = pd.concat([df1, df2['Value2']], axis=1)

# 打印合并后的DataFrame
print(df_merged)
