import pandas as pd

class preprocessing:
    @staticmethod
    def data_read():
        # loading dataset using pandas
        df_fake = pd.read_csv("Dataset/Fake.csv", engine='python', on_bad_lines='skip', encoding='utf-8')
        df_true = pd.read_csv("Dataset/True.csv", engine='python', on_bad_lines='skip', encoding='utf-8')
        
        # adding label 
        df_true['label'] = 1
        df_fake['label'] = 0
        
        #combining the dataset
        df = pd.concat([df_fake, df_true],axis=0).reset_index(drop=True)
        
        #Reseting the index
        df = df.sample(frac=1,random_state=42).reset_index(drop=True)
        
        #Creating the new columns combining title and text column
        df['contents'] = df['title'] + '\n' + df['text']
        
        #remvoing unnecessary columns
        df.drop(['title','text','date','subject'],axis=1,inplace=True)
        
        return df
    
    