import numpy as np
import NRCS.constants as const

def set_nan(data):
    np.ma.set_fill_value(data, 1e3)
    data = data.filled()
    data[data == 1e3] = np.nan
    return data[0, :, :]

def nearest_nan(data, lat, lon):
    boundary = 100*np.ones([data.shape[0]+2, data.shape[1]+2])
    boundary[1:boundary.shape[0]-1, 1:boundary.shape[1]-1] = data
    [x,y] = np.where(np.isnan(boundary))
    boundary[0, :] = np.nan
    boundary[boundary.shape[0]-1, :] = np.nan
    boundary[:, 0] = np.nan
    boundary[:, boundary.shape[1]-1] = np.nan
    data = boundary
    # set lat
    boundary = 100*np.ones([lat.shape[0]+2, lat.shape[1]+2])
    boundary[1:boundary.shape[0]-1, 1:boundary.shape[1]-1] = lat
    boundary[0, :] = np.nan
    boundary[boundary.shape[0]-1, :] = np.nan
    boundary[:, 0] = np.nan
    boundary[:, boundary.shape[1]-1] = np.nan
    lat = boundary
    # set lon
    boundary=100*np.ones([lon.shape[0]+2,lon.shape[1]+2])
    boundary[1:boundary.shape[0]-1,1:boundary.shape[1]-1]=lon
    boundary[0,:]=np.nan
    boundary[boundary.shape[0]-1,:]=np.nan
    boundary[:,0]=np.nan
    boundary[:,boundary.shape[1]-1]=np.nan
    lon = boundary
    for i in range(0, len(x)):
        position = np.array([[lat[x[i] - 1, y[i] - 1], lon[x[i] - 1, y[i] - 1]],
        [lat[x[i] - 1, y[i]], lon[x[i] - 1, y[i]]],
        [lat[x[i] - 1, y[i] + 1], lon[x[i] - 1, y[i] + 1]],
        [lat[x[i], y[i] - 1], lon[x[i], y[i] - 1]],
        [lat[x[i], y[i] + 1], lon[x[i], y[i] + 1]],
        [lat[x[i] + 1, y[i] - 1], lon[x[i] + 1, y[i] - 1]],
        [lat[x[i] + 1, y[i]], lon[x[i] + 1, y[i]]],
        [lat[x[i] + 1, y[i] + 1], lon[x[i] + 1, y[i] + 1]]])
        neighbor = np.array([[data[x[i]-1, y[i]-1]],
            [data[x[i] - 1, y[i]]],
            [data[x[i] - 1, y[i] + 1]],
            [data[x[i], y[i] - 1]],
            [data[x[i], y[i] + 1]],
            [data[x[i] + 1, y[i] - 1]],
            [data[x[i] + 1, y[i]]],
            [data[x[i] + 1, y[i] + 1]]])
        ind = np.where(np.isnan(neighbor))
        ind = ind[0]
        position = np.delete(position, ind, 0)
        neighbor = np.delete(neighbor, ind, 0)
# harversin
        dlat = (lat[x[i], y[i]] - position[:, 0]) * np.pi / 180
        dlon = (lon[x[i], y[i]] - position[:, 1]) * np.pi / 180
        dist = 6371.0 * 2 * np.arcsin(
            np.sqrt(np.sin(dlat / 2) ** 2 + np.cos(lat[x[i],y[i]] * np.pi / 180) * np.cos(position[:, 0] * np.pi / 180) * np.sin(dlon / 2) ** 2))

        if len(dist) != 0:
            coe = np.where(dist == min(dist))
            coe = coe[0]
            data[x[i], y[i]] = neighbor[coe, :]
        else:
            data[x[i], y[i]] = 0
    data_new = data[1:data.shape[0]-1, 1:data.shape[1]-1]
    return data_new

def mean_zeros(data):
    boundary = 100*np.ones([data.shape[0]+2,data.shape[1]+2])
    boundary[1:boundary.shape[0]-1, 1:boundary.shape[1]-1] = data
    [x, y] = np.where(boundary == 0)
    boundary[0, :]=np.nan
    boundary[boundary.shape[0]-1, :] = np.nan
    boundary[:, 0]=np.nan
    boundary[:, boundary.shape[1]-1] = np.nan
    data = boundary
    for i in range(0, len(x)):
        neighbor = np.array([[data[x[i]-1, y[i]-1]],
            [data[x[i]-1, y[i]]],
            [data[x[i]-1, y[i]+1]],
            [data[x[i], y[i]-1]],
            [data[x[i], y[i]+1]],
            [data[x[i] + 1, y[i] - 1]],
            [data[x[i] + 1, y[i]]],
            [data[x[i]+1, y[i]+1]]])
        ind = np.where(np.isnan(neighbor))
        ind = ind[0]
        neighbor = np.delete(neighbor, ind, 0)
        data[x[i], y[i]] = np.mean(neighbor)
    data_new = data[1:data.shape[0]-1, 1:data.shape[1]-1]
    return data_new

def del2(M):
    ndim = 2
    loc = np.array([range(0, M.shape[0]), range(0, M.shape[1])])
    rflag = 0
    v = np.zeros([M.shape[0], M.shape[1]])
    perm = np.array([1, 0])
    for i in range(0, 2):
        n = M.shape[0]
        p = M.shape[1]
        x = loc[i]
        h = np.diff(x)
        h = h.reshape(h.shape[0],1)
        g = np.zeros([M.shape[0], M.shape[1]])
        if n > 2:
            g[1:n-1, :] = (np.diff(M[1:n, :], axis=0)/h[1:n-1] - np.diff(M[0:n-1, :], axis=0)/h[0:n-2])\
                          / (h[1:n-1] + h[0:n-2])
        if n > 3:
            g[0, :] = g[1, :]*(h[0]+h[1])/h[1] - g[2, :]*h[0]/h[1]
            g[n-1, :] = -g[n-3, :]*h[n-2]/h[n-3] + g[n-2, :]*(h[n-2]+h[n-3])/h[n-3]
        elif n == 3:
            g[0, :] = g[1, :]
            g[n-1, :] = g[1, :]
        else:
            g[0, :] = 0
            g[n-1, :] = 0
        if i == 0:
            v = v + g
        else:
            v = v + np.transpose(g)

        M = np.transpose(M)
    v = v/2
    return v

def vorticity(sst, lat, lon, K):

    # interpolate data
    sst = set_nan(sst)
    sst = nearest_nan(sst, lat, lon)
    sst = mean_zeros(sst)

    T = np.fft.fft2(sst)
    z = -10 ** (-6)
    n0 = const.nb
    stream = const.g * const.alpha * T * np.exp(n0 * K * z) / (const.f * const.nb * K)
    stream[np.isinf(stream)] = 0
    stream[np.isnan(stream)] = 0
    vor = 4 * del2(np.fft.ifft2(stream)) / 1000 ** 2
    return vor