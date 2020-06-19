from lemipy import Survey, list_files
import multiprocessing
from lemipy.filters import despike
# lemipy.lemi.TIMEZONE = 'Australia/Adelaide'
import export_options
from datetime import datetime as dt

survey = Survey(directory='NVP', remote='C5', survey_data='survey_data.csv', channels=['Bx','By','Ex','Ey'])

def test():

    site = survey.sites['B8']
    site.load_file(0)
    opt = getattr(export_options, "B8")
    site.in_memory.filter(opt['filter'], in_place=True)
    Bx = site.in_memory.data.Bx
    windows, wavelet = despike(Bx,[100,200],1000,[10,30],40)
    return Bx, windows, wavelet

if __name__ == '__main__':
    start = dt.now()

    multiprocessing.freeze_support()  
    exclude = ['A1','A2','A3','A4','A5','A6','A7','A8','A9',
    'B1','B2','B3','B4','B5','B6','B7','B8','B9',
    'C1','C2','C3','C4','C5 [REMOTE]','C5b [REMOTE]','C6','C6 [REPEAT]','C7','C8','C9'
    'D3','D5','D6','E1','E6','F2','F4']

    for_export = {name: site for name, site in survey.sites.items() if name not in exclude}

    for name, site in for_export.items():
        site.batch_export()
        # print(name)
        # if c >= 1:
        #     x = list_files(os.path.join(survey.directory,sites[c-1]),'.txt','')
        # c+=1
    
    print('All {} sites finished in {}'.format(len(for_export),dt.now()-start))



