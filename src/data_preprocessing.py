import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



def load_data(filepath):
    return pd.read_csv(filepath)


def preprocess_data(df):
    df.drop(columns=['Unnamed: 0'], inplace=True)
    df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)
    df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)
    df['Price'] = df['Price'].astype(int)

    df['Touchscreen']= df['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)
    df['IPS']=df['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)

    new = df['ScreenResolution'].str.split('x',n=1,expand=True)
    df['X_res']= new[0]
    df['Y_res']= new[1]
    df['X_res']= df['X_res'].str.replace(',','').str.findall(r'(\d+\.?\d+)').apply(lambda x:x[0])
    df['X_res']=df['X_res'].astype('int32')
    df['Y_res']=df['Y_res'].astype('int32')
    df['ppi']=(((df['X_res']**2) + (df['Y_res']**2))**0.5/df['Inches']).astype('float32')
    df.drop(columns=['ScreenResolution','Inches','X_res','Y_res'],inplace= True)

    df['Cpu_Name']=df['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))

    def fetch_processor(text):
        if text == 'Intel Core i3' or text == 'Intel Core i5' or text == 'Intel Core i7':
            return text
        else:
            if text.split()[0] == 'Intel':
                return 'Other Intel Processor'
            else:
                return 'AMD Processor'
            
    df['Cpu_Brand']=df['Cpu_Name'].apply(fetch_processor)

    df.drop(columns=['Cpu','Cpu_Name'],inplace=True)

    # breaking Memory column into understandable columns as feature

    df['Memory']=df['Memory'].astype(str).replace('\.0','',regex=True)
    df['Memory']=df['Memory'].str.replace('GB','')
    df['Memory']=df['Memory'].str.replace('TB','000')
    new= df['Memory'].str.split("+",n=1,expand=True)

    df['first']=new[0]
    df['first']=df['first'].str.strip()

    df['second']=new[1]

    df['Layer1HDD'] = df['first'].apply(lambda x:1 if 'HDD' in x else 0)
    df['Layer1SSD'] = df['first'].apply(lambda x:1 if 'SSD' in x else 0)
    df['Layer1Hybrid'] = df['first'].apply(lambda x:1 if 'Hybrid' in x else 0)
    df['Layer1Flash_Storage'] = df['first'].apply(lambda x:1 if 'Flash_Storage' in x else 0)

    df['first']= df['first'].str.replace(r'\D','',regex= True)

    df['second'].fillna('0', inplace=True)

    df['Layer2HDD'] = df['second'].apply(lambda x:1 if 'HDD' in x else 0)
    df['Layer2SSD'] = df['second'].apply(lambda x:1 if 'SSD' in x else 0)
    df['Layer2Hybrid'] = df['second'].apply(lambda x:1 if 'Hybrid' in x else 0)
    df['Layer2Flash_Storage'] = df['first'].apply(lambda x:1 if 'Flash_Storage' in x else 0)

    df['second']= df['second'].str.replace(r'\D','',regex=True)

    df['first'] = df['first'].replace('', '0').astype(int)
    df['second'] = df['second'].replace('', '0').astype(int)


    df['HDD']=(df['first']*df['Layer1HDD']+df['second']*df['Layer2HDD'])
    df['SSD']=(df['first']*df['Layer1SSD']+df['second']*df['Layer2SSD'])
    df['Hybrid']=(df['first']*df['Layer1Hybrid']+df['second']*df['Layer2Hybrid'])
    df['Flash_Storage']=(df['first']*df['Layer1Flash_Storage']+df['second']*df['Layer2Flash_Storage'])

    df.drop(columns=['first','second','Layer1HDD','Layer1SSD','Layer1Hybrid','Layer1Flash_Storage',
                    'Layer2HDD','Layer2SSD','Layer2Hybrid','Layer2Flash_Storage'],inplace=True)
    
    
    df.drop(columns=['Memory','Hybrid','Flash_Storage'],inplace=True)

    df['Gpu']=df['Gpu'].apply(lambda x:x.split()[0])
    df= df[df['Gpu']!='ARM']

    def cat_os(inp):
        if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 S':
            return 'Windows'
        elif inp == 'macOS' or inp== 'Mac OS X':
            return 'Mac'
        else:
            return 'Others/No OS/Linux'
        
    df['os']= df['OpSys'].apply(cat_os)
    df.drop(columns=['OpSys'],inplace=True)

    # defining X and y variables
    X= df.drop(columns=['Price'])
    y= np.log(df['Price'])

    return X, y