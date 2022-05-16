import pandas as pd
import matplotlib.pyplot as plt

from pandas_profiling import ProfileReport
import seaborn as sns

PATH = '~/Downloads/heart_cleveland_upload.csv'


def main():
    df = pd.read_csv(PATH)

    profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
    profile.to_file('./notebooks/heart_disease_cleveland/report/pandas_profiling_report.html')

    cat_feat_sample = ['sex', 'cp', 'exang']
    num_feat = ['age', 'trestbps', 'oldpeak', 'thalach', 'chol']

    fig, axs = plt.subplots(nrows=len(num_feat), ncols=len(cat_feat_sample), figsize=(15, 20))
    for ncol, cf in enumerate(cat_feat_sample):
        for nrow, nf in enumerate(num_feat):
            sns.histplot(data=df, x=nf, hue=cf, ax=axs[nrow][ncol])
    fig.savefig('./notebooks/heart_disease_cleveland/report/raw_files/chart.png')

    html_cat_stat = ''
    for cf in cat_feat_sample:
        html_cat_stat += '\n' + f'<h2>{cf} stat</h2>'
        html_cat_stat += '\n' + df.groupby('sex').count().reset_index().to_html()
    report_html = f'''
        <html>
            <head>
                <title>Some report</title>
            </head>
            <body>
                <h1>Distribution with respect some categorical feature</h1>
                <p>list of categorical features: {', '.join(cat_feat_sample)}</p>
                <img src='./notebooks/heart_disease_cleveland/report/raw_files/chart.png' width="700">
                {html_cat_stat[1:]}
            </body>
        </html>
        '''
    with open('./notebooks/heart_disease_cleveland/report/my_report.html', 'w') as f:
        f.write(report_html)


if __name__ == '__main__':
    main()
