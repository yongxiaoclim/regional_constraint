import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import xarray as xr
import shapely
import sys
import cartopy.crs as ccrs
import rasterio
import regionmask
import matplotlib.pyplot as plt
import statsmodels.api as sm
from netCDF4 import Dataset as open_ncfile
from test import model_with_ensembles
from test import weighted_quantile
from test import find_nearest
from scipy.stats import t
from scipy.stats import norm
from scipy import stats
import pandas as pd
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable


def draw_bs_sample(data):
    """Draw a bootstrap sample from a 1D data set."""
    return np.random.choice(data, size=len(data))

def draw_bs_pairs(x, y):
    """Draw a pairs bootstrap sample."""
    inds = np.arange(len(x))
    bs_inds = draw_bs_sample(inds)

    return x[bs_inds], y[bs_inds]


xr.set_options(display_style="text")
np.set_printoptions(edgeitems=2)
regionmask.defined_regions.ar6
ar6_all = regionmask.defined_regions.ar6.all
con_reg=[0,1, 2, 3,4,5,16,17,18,19,28,29,30,31,32,33,34,35]

##obs_zhai_pool=np.random.normal(loc=-1.28, scale=0.19, size=200)
##obs_zhai_pool=np.arange(-1.5, -1.3, 0.01)
##obs_sha_pool=np.arange(43, 48, 0.1)

model_name= ["GISS-E2-H","GISS-E2-H-CC",  "MRI-ESM1",
"BCC-CSM1-1", "FGOALS-g2","IPSL-CM5A-MR","IPSL-CM5B-LR","IPSL-CM5A-LR",
"CanESM2","CCSM4",
           "CSIRO-Mk3-6-0",
              "MPI-ESM-P",
             "NorESM1-M","NorESM1-ME"]
             

path= '/kenes/user/liang/data/tas_change_spatial/year/fu/cmip5/change/ens/'
# pathtxx= '/kenes/user/liang/data/tas_change_spatial/'
path_metric='/kenes/user/liang/code/regional_metric/metric_3cal/cmip5/'

times=100
number=100

tas=np.arange(np.size(model_name)*180*360).reshape(np.size(model_name),180,360)
tas= tas.astype(np.float64)

untas=np.arange(np.size(model_name)*180*360).reshape(np.size(model_name),180,360)
untas= untas.astype(np.float64)

untxx=np.empty(np.shape(untas),float)

hfls_me=np.empty(np.shape(untas),float)
hfss_me=np.empty(np.shape(untas),float)
huss_me=np.empty(np.shape(untas),float)
pr_me=np.empty(np.shape(untas),float)
psl_me=np.empty(np.shape(untas),float)
rlus_me=np.empty(np.shape(untas),float)
rsds_me=np.empty(np.shape(untas),float)
tas_me=np.empty(np.shape(untas),float)

cor_hfls=np.arange(times*number*np.size(con_reg)).reshape(times,number,np.size(con_reg))
cor_hfls= cor_hfls.astype(np.float64)
cor_hfss=np.empty(np.shape(cor_hfls),float)
cor_huss=np.empty(np.shape(cor_hfls),float)
cor_pr=np.empty(np.shape(cor_hfls),float)
cor_psl=np.empty(np.shape(cor_hfls),float)
cor_rlus=np.empty(np.shape(cor_hfls),float)
cor_rsds=np.empty(np.shape(cor_hfls),float)
cor_zh=np.empty(np.shape(cor_hfls),float)
cor_bcs=np.empty(np.shape(cor_hfls),float)
cor_tas=np.empty(np.shape(cor_hfls),float)
cor_gsat=np.empty(np.shape(cor_hfls),float)

for r in range(times):
    print(r)
    zh_r=[]
    sw_r=[]
    GSAT_r=[]
 
    for i in range(len(model_name)):
        
        gsat_all=np.loadtxt('/kenes/user/liang/code/regional_metric/metric_3cal/cmip5/tas/ens/GSAT_trend/'+model_name[i]+'.txt',delimiter='\t')
        if np.size(gsat_all)>1:
            GSAT_r=np.append(GSAT_r,np.random.choice(gsat_all))
        else:
            GSAT_r=np.append(GSAT_r,gsat_all)

        zhai=open_ncfile('/kenes/user/liang/code/regional_metric/metric_3cal/cmip5/MBLC/'+model_name[i]+'_cloud.nc')
        zh=zhai.variables['lcc_sst_g']
        zh_r=np.append(zh_r,np.random.choice(zh))

        swcre=open_ncfile('/kenes/user/liang/code/regional_metric/metric_3cal/cmip5/BCS/'+model_name[i]+'_cloud.nc')
        sw=swcre.variables['index']
        sw_r=np.append(sw_r,np.random.choice(sw)*100)
     
        ds = xr.open_dataset(path+model_name[i]+"_tas_year_2006_2081.nc")
        size=np.size(ds.tas_all[:,0,0])
        num=np.random.choice(range(size))
        untas[i,:,:]=ds.tas_all[num,:,:]

        ds = xr.open_dataset(path_metric+'/hfls/ens/'+model_name[i]+"_tasmax_yealy_his.nc")
        size=np.size(ds.trend[:,0,0])
        num=np.random.choice(range(size))
        hfls_me[i,:,:]=ds.trend[num,:,:]

        ds = xr.open_dataset(path_metric+'/hfss/ens/'+model_name[i]+"_tasmax_yealy_his.nc")
        size=np.size(ds.trend[:,0,0])
        num=np.random.choice(range(size))
        hfss_me[i,:,:]=ds.trend[num,:,:]

        ds = xr.open_dataset(path_metric+'/huss/ens/'+model_name[i]+"_tasmax_yealy_his.nc")
        size=np.size(ds.trend[:,0,0])
        num=np.random.choice(range(size))
        huss_me[i,:,:]=ds.trend[num,:,:]

        ds = xr.open_dataset(path_metric+'/pr/ens/'+model_name[i]+"_tasmax_yealy_his.nc")
        size=np.size(ds.trend[:,0,0])
        num=np.random.choice(range(size))
        pr_me[i,:,:]=ds.trend[num,:,:]

        ds = xr.open_dataset(path_metric+'/psl/ens/'+model_name[i]+"_tasmax_yealy_his.nc")
        size=np.size(ds.trend[:,0,0])
        num=np.random.choice(range(size))
        psl_me[i,:,:]=ds.trend[num,:,:]

        ds = xr.open_dataset(path_metric+'/rlus/ens/'+model_name[i]+"_tasmax_yealy_his.nc")
        size=np.size(ds.trend[:,0,0])
        num=np.random.choice(range(size))
        rlus_me[i,:,:]=ds.trend[num,:,:]

        ds = xr.open_dataset(path_metric+'/rsds/ens/'+model_name[i]+"_tasmax_yealy_his.nc")
        size=np.size(ds.trend[:,0,0])
        num=np.random.choice(range(size))
        rsds_me[i,:,:]=ds.trend[num,:,:]

        ds = xr.open_dataset(path_metric+'/tas/ens/'+model_name[i]+"_tasmax_yealy_his.nc")
        size=np.size(ds.trend[:,0,0])
        num=np.random.choice(range(size))
        tas_me[i,:,:]=ds.trend[num,:,:]



    tas_region=xr.DataArray(untas,dims=['ens','lat', 'lon'],
               coords={'ens':model_name,'lat': ds.lat,'lon': ds.lon},)
    weights = np.cos(np.deg2rad(tas_region.lat))
    mask_tas = ar6_all.mask_3D(tas_region)
    tas_regional = tas_region.weighted(mask_tas* weights).mean(dim=("lat", "lon"))


    hfls_region=xr.DataArray(hfls_me,dims=['ens','lat', 'lon'],
               coords={'ens':model_name,'lat': ds.lat,'lon': ds.lon},)
    mask_hfls = ar6_all.mask_3D(hfls_region)
    hfls_regional = hfls_region.weighted(mask_hfls* weights).mean(dim=("lat", "lon"))


    hfss_region=xr.DataArray(hfss_me,dims=['ens','lat', 'lon'],
               coords={'ens':model_name,'lat': ds.lat,'lon': ds.lon},)
    mask_hfss = ar6_all.mask_3D(hfss_region)
    hfss_regional = hfss_region.weighted(mask_hfss* weights).mean(dim=("lat", "lon"))


    huss_region=xr.DataArray(huss_me,dims=['ens','lat', 'lon'],
               coords={'ens':model_name,'lat': ds.lat,'lon': ds.lon},)
    mask_huss = ar6_all.mask_3D(huss_region)
    huss_regional = huss_region.weighted(mask_huss* weights).mean(dim=("lat", "lon"))


    pr_region=xr.DataArray(pr_me,dims=['ens','lat', 'lon'],
               coords={'ens':model_name,'lat': ds.lat,'lon': ds.lon},)
    mask_pr = ar6_all.mask_3D(pr_region)
    pr_regional = pr_region.weighted(mask_pr* weights).mean(dim=("lat", "lon"))


    psl_region=xr.DataArray(psl_me,dims=['ens','lat', 'lon'],
               coords={'ens':model_name,'lat': ds.lat,'lon': ds.lon},)
    mask_psl = ar6_all.mask_3D(psl_region)
    psl_regional = psl_region.weighted(mask_psl* weights).mean(dim=("lat", "lon"))


    rlus_region=xr.DataArray(rlus_me,dims=['ens','lat', 'lon'],
               coords={'ens':model_name,'lat': ds.lat,'lon': ds.lon},)
    mask_rlus = ar6_all.mask_3D(rlus_region)
    rlus_regional = rlus_region.weighted(mask_rlus* weights).mean(dim=("lat", "lon"))


    rsds_region=xr.DataArray(rsds_me,dims=['ens','lat', 'lon'],
               coords={'ens':model_name,'lat': ds.lat,'lon': ds.lon},)
    mask_rsds = ar6_all.mask_3D(rsds_region)
    rsds_regional = rsds_region.weighted(mask_rsds* weights).mean(dim=("lat", "lon"))

    htas_region=xr.DataArray(tas_me,dims=['ens','lat', 'lon'],
               coords={'ens':model_name,'lat': ds.lat,'lon': ds.lon},)
    mask_htas = ar6_all.mask_3D(htas_region)
    htas_regional = htas_region.weighted(mask_htas* weights).mean(dim=("lat", "lon"))

##    print(np.shape(rsds_regional))
##    print(np.shape(tas_regional[0]))
    
    for re in range(len(con_reg)):
    	for tm in range(number):
            # print(np.shape((draw_bs_pairs(GSAT_r,tas_regional[:,re]))))
            aa=draw_bs_pairs(GSAT_r,tas_regional[:,re])
            cor_gsat[r,tm,re]=stats.pearsonr(aa[0],aa[1])[0]
            aa=draw_bs_pairs(zh_r,tas_regional[:,re])
            cor_zh[r,tm,re]=stats.pearsonr(aa[0],aa[1])[0]
            aa=draw_bs_pairs(sw_r,tas_regional[:,re])
            cor_bcs[r,tm,re]=stats.pearsonr(aa[0],aa[1])[0]
            aa=draw_bs_pairs(hfls_regional[:,re],tas_regional[:,re])
            cor_hfls[r,tm,re]=stats.pearsonr(aa[0],aa[1])[0]
            aa=draw_bs_pairs(hfss_regional[:,re],tas_regional[:,re])
            cor_hfss[r,tm,re]=stats.pearsonr(aa[0],aa[1])[0]
            aa=draw_bs_pairs(huss_regional[:,re],tas_regional[:,re])
            cor_huss[r,tm,re]=stats.pearsonr(aa[0],aa[1])[0]
            aa=draw_bs_pairs(pr_regional[:,re],tas_regional[:,re])
            cor_pr[r,tm,re]=stats.pearsonr(aa[0],aa[1])[0]
            aa=draw_bs_pairs(psl_regional[:,re],tas_regional[:,re])
            cor_psl[r,tm,re]=stats.pearsonr(aa[0],aa[1])[0]
            aa=draw_bs_pairs(rlus_regional[:,re],tas_regional[:,re])
            cor_rlus[r,tm,re]=stats.pearsonr(aa[0],aa[1])[0]
            aa=draw_bs_pairs(rsds_regional[:,re],tas_regional[:,re])
            cor_rsds[r,tm,re]=stats.pearsonr(aa[0],aa[1])[0]
            aa=draw_bs_pairs(htas_regional[:,re],tas_regional[:,re])
            cor_tas[r,tm,re]=stats.pearsonr(aa[0],aa[1])[0]
                  
np.save(file="cor_hfls.npy", arr=cor_hfls)
np.save(file="cor_hfss.npy", arr=cor_hfss)
np.save(file="cor_huss.npy", arr=cor_huss)
np.save(file="cor_pr.npy", arr=cor_pr)
np.save(file="cor_psl.npy", arr=cor_psl)
np.save(file="cor_rlus.npy", arr=cor_rlus)
np.save(file="cor_rsds.npy", arr=cor_rsds)
np.save(file="cor_htas.npy", arr=cor_tas)
np.save(file="cor_gsat.npy", arr=cor_gsat)


np.save(file="cor_zh.npy", arr=cor_zh)
np.save(file="cor_bcs.npy", arr=cor_bcs)

        
