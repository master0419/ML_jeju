import pandas as pd

xls = pd.ExcelFile(r'ref/[테크핀레이팅스]7개 데이터상품_테이블명세서_V4.0_241209.xlsx')

for sheet in ['0.제공테이블', '1.월별_매출정보', '2.월별_매입정보', '3.거래처별매출채권정보']:
    df = pd.read_excel(xls, sheet_name=sheet)
    if df.empty:
        continue
    print(f'\n=== {sheet} ===')
    print(f'Shape: {df.shape}')
    sep = ' | '
    for i, row in df.iterrows():
        vals = [str(v) for v in row.values if str(v) != 'nan']
        if vals:
            print(f'  {i}: {sep.join(vals)}')
