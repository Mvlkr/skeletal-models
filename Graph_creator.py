import os
import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.axis import Axis
from skimage.transform import resize


def dict_generator(players: list[int]) -> dict:
    dict_gr = {}
    for p in players:
        dict_gr[p] = {
        'difference': {
            'left_x': [],
            'left_y': [],
            'right_x': [],
            'right_y': [],
            'left_x_n': [],
            'left_y_n': [],
            'right_x_n': [],
            'right_y_n': []

        },
        'coords': {
            'left_x': [],
            'left_y': [],
            'right_x': [],
            'right_y': []
        },
        'legs': { # if the leg in air
            'left': [],
            'right': [],
        },
        'bird_view': {
            'left_x': [],
            'left_y': [],
            'right_x': [],
            'right_y': []
        },
        'frames': []
    }
    return dict_gr


class Graph_creator:
    def __init__(self, H, tracks: list[int], file: dict, time_start: int = 0, time_stop: int = 120, frames_per_s: int = 30, jump_pix: int = 10) -> None:
        self.j = file
        self.frames_start = time_start*frames_per_s
        self.frames_stop = time_stop*frames_per_s
        self.frames_per_s = frames_per_s
        self.d = dict_generator(tracks)
        self.fl = False
        self.H = H
        self.jump_pix = jump_pix


    def __find_min_position(self):
        for fr in range(self.frames_start, len(self.j['FrameSequences'][0]['Frames'])):
            amount_objects = len(self.j['FrameSequences'][0]['Frames'][fr]['DetectedObjects'])
            for obj in range(amount_objects):
                if (self.j['FrameSequences'][0]['Frames'][fr]['DetectedObjects'][obj]['Track'] in self.d.keys()):
                    pl = self.j['FrameSequences'][0]['Frames'][fr]['DetectedObjects'][obj]['Track']
                    self.d[pl]['frames'].append(fr)
                    
                    l_49 = self.j['FrameSequences'][0]['Frames'][fr]['DetectedObjects'][obj]['Bones'][49]
                    l_45 = self.j['FrameSequences'][0]['Frames'][fr]['DetectedObjects'][obj]['Bones'][45]
                    l_47 = self.j['FrameSequences'][0]['Frames'][fr]['DetectedObjects'][obj]['Bones'][47]
                    
                    if l_49 == l_45 == l_47 == -1:
                        min_l_x = None
                        min_l_y = None
                        self.d[pl]['coords']['left_x'].append(None)
                        self.d[pl]['coords']['left_y'].append(None)
                    else:
                        l_y = np.array([l_45, l_47, l_49])
                        points_l = np.where(l_y != -1, l_y, max(l_y))
                        min_l_y = max(points_l)
                        min_l_x = self.j['FrameSequences'][0]['Frames'][fr]['DetectedObjects'][obj]['Bones'][44 + 2*np.argmin(points_l)]
                        
                        if (min_l_x == -1):
                            min_l_x = None
                            min_l_y = None
                            
                        self.d[pl]['coords']['left_x'].append(min_l_x)
                        self.d[pl]['coords']['left_y'].append(min_l_y)
                
                    r_39 = self.j['FrameSequences'][0]['Frames'][fr]['DetectedObjects'][obj]['Bones'][39]
                    r_41 = self.j['FrameSequences'][0]['Frames'][fr]['DetectedObjects'][obj]['Bones'][41]
                    r_43 = self.j['FrameSequences'][0]['Frames'][fr]['DetectedObjects'][obj]['Bones'][43]
                    
                    if r_39 == r_41 == r_43 == -1:
                        min_r_x = None
                        min_r_y = None
                        self.d[pl]['coords']['right_x'].append(None)
                        self.d[pl]['coords']['right_y'].append(None)
                        
                    else:
                        r_y = np.array([r_39, r_41, r_43])
                        points_r = np.where(r_y != -1, r_y, max(r_y))
                        min_r_y = max(points_r)
                        min_r_x = self.j['FrameSequences'][0]['Frames'][fr]['DetectedObjects'][obj]['Bones'][38 + 2*np.argmin(points_r)]
                        
                        if (min_r_x == -1):
                            min_r_x = None
                            min_r_y = None
                    
                        self.d[pl]['coords']['right_x'].append(min_r_x)
                        self.d[pl]['coords']['right_y'].append(min_r_y)

                    ## разность:
                    central_x = self.j['FrameSequences'][0]['Frames'][fr]['DetectedObjects'][obj]['Bones'][16]
                    central_y = self.j['FrameSequences'][0]['Frames'][fr]['DetectedObjects'][obj]['Bones'][17]

                    if central_x == -1 or min_l_x == None:
                        self.d[pl]['difference']['left_x'].append(None)
                        self.d[pl]['difference']['left_x_n'].append(None)

                    else:
                        self.d[pl]['difference']['left_x'].append(central_x - min_l_x)
                        self.d[pl]['difference']['left_x_n'].append((min_l_x- central_x)**2)

                    if central_x == -1 or min_r_x == None:
                        self.d[pl]['difference']['right_x'].append(None)
                        self.d[pl]['difference']['right_x_n'].append(None)
                    else:
                        self.d[pl]['difference']['right_x'].append(central_x - min_r_x)
                        self.d[pl]['difference']['right_x_n'].append((min_r_x- central_x)**2)

                    if central_y == -1 or min_l_y == None:
                        self.d[pl]['difference']['left_y'].append(None)
                        self.d[pl]['difference']['left_y_n'].append(None)
                    else:
                        self.d[pl]['difference']['left_y'].append(central_y - min_l_y)
                        self.d[pl]['difference']['left_y_n'].append((min_l_y- central_y)**2)
                    
                    if central_y == -1 or min_r_y == None:
                        self.d[pl]['difference']['right_y'].append(None)
                        self.d[pl]['difference']['right_y_n'].append(None)
                    else:
                        self.d[pl]['difference']['right_y'].append(central_y - min_r_y)
                        self.d[pl]['difference']['right_y_n'].append((min_r_y- central_y)**2)

                    
                    y = self.j['FrameSequences'][0]['Frames'][fr]['DetectedObjects'][obj]['BBox'][3] # смотрим на y относительно bbox

                    min_arg = np.argwhere(np.array(self.d[pl]['frames']) == fr)
                    if len(self.d[pl]['frames']) > 1:
                        fr_p = self.d[pl]['frames'][-2]
                        pr_arg = np.argwhere(np.array(self.d[pl]['frames']) == fr_p)
                    
                    # локальныe минимумы для игрока
                    min_l = self.d[pl]['coords']['left_y'][min_arg[0][0]] 
                    min_r = self.d[pl]['coords']['right_y'][min_arg[0][0]] 

                    if len(self.d[pl]['frames']) == 1 :
                        if (min_l != None):
                            if  abs(min_l - y) >= 40:
                                self.d[pl]['legs']['left'].append(2)
                            else:
                                self.d[pl]['legs']['left'].append(1)
                        if (min_r != None):
                            if  abs(min_r - y) >= 40:
                                self.d[pl]['legs']['right'].append(4)
                            else:
                                self.d[pl]['legs']['right'].append(3)

                    
                    # для построения циклограммы
                    # если у нас не существует предыдущего положения ног, то мы определяем положения по оторванности от рамки?????
                    # если же положение существует, то если расстояние между текущеми координатами и предыдущими больше порога,
                    #  то смена положения, иначе остаемся в исходном
                    if (min_l != None) and len(self.d[pl]['frames']) >= 2:
                        if self.d[pl]['legs']['left'][-1] == None:
                            new_pos_l = None
                        else:
                            new_pos_l = (self.d[pl]['legs']['left'][-1] % 2) + 1
                        if self.d[pl]['coords']['left_y'][pr_arg[0][0]] == None or new_pos_l == None:
                            if  abs(min_l - y) >= 15:
                                self.d[pl]['legs']['left'].append(2)
                            else:
                                self.d[pl]['legs']['left'].append(1)
                        else:
                            if abs(self.d[pl]['coords']['left_y'][pr_arg[0][0]] - min_l) > self.jump_pix and abs(min_l - y) >= self.jump_pix: # change position
                                self.d[pl]['legs']['left'].append(new_pos_l)
                            else:
                                self.d[pl]['legs']['left'].append(self.d[pl]['legs']['left'][-1])
                    else:
                        if (min_l == None):
                            self.d[pl]['legs']['left'].append(None)
                        
                    if (min_r != None) and len(self.d[pl]['frames']) >= 2:
                        if self.d[pl]['legs']['right'][-1] == None:
                            new_pos_r = None
                        else:
                            new_pos_r = (self.d[pl]['legs']['right'][-1] % 2) + 3
                        if self.d[pl]['coords']['right_y'][pr_arg[0][0]] == None:
                            if  abs(min_r - y) >= 15:
                                self.d[pl]['legs']['right'].append(4)
                            else:
                                self.d[pl]['legs']['right'].append(3)
                        else:
                            if abs(self.d[pl]['coords']['right_y'][pr_arg[0][0]] - min_r) > self.jump_pix and abs(min_r - y) >= self.jump_pix:
                                self.d[pl]['legs']['right'].append(new_pos_r)
                            else:
                                self.d[pl]['legs']['right'].append(self.d[pl]['legs']['right'][-1])
                    else:
                        if (min_r == None):
                            self.d[pl]['legs']['right'].append(None)  
        self.fl = True  


    def __checker(self):
        if self.fl:
            pass
        else:
            self.__find_min_position()


    def __save(self, filename: str):
        folder = '/for_data/'
        abpath = os.path.abspath(__file__)
        path = os.path.split(abpath)[0]
        directory = path + folder

        if not os.path.exists(directory):
            os.makedirs(directory)

        plt.savefig(directory+filename, bbox_inches='tight')


    def __frames_to_sec(self, pl: int = 1) -> list[int]:
        sec = int((self.d[pl]['frames'][0] / self.frames_per_s))
        rest = self.d[pl]['frames'][0] % self.frames_per_s
        x_ticks = [sec-1]
        for i in range(1, len(self.d[pl]['frames'])):
            if self.d[pl]['frames'][i] - self.d[pl]['frames'][i - 1] == 1:
               if (len(x_ticks) + rest) % self.frames_per_s == 0:
                   sec += 1
                   x_ticks.append(sec)
               else:
                   x_ticks.append(sec)
            else:
                omitted = self.d[pl]['frames'][i] - self.d[pl]['frames'][i - 1]
                rest += omitted % self.frames_per_s
                if int(omitted / self.frames_per_s) > 0:
                    sec += 1
                x_ticks.append(sec)

        return np.unique(x_ticks)
    
    def creating_data(self, start, end, name, which):
        st = (self.frames_per_s/1000) * (start*1000)
        en = (self.frames_per_s/1000) * (end*1000)
        datas = []
        for pl in self.d.keys():
            l_x_n = np.nansum(np.array(self.d[pl]['difference']['left_x_n'], dtype=float) / len(self.d[pl]['difference']['left_x_n']))
            l_y_n = np.nansum(np.array(self.d[pl]['difference']['left_y_n'], dtype=float) / len(self.d[pl]['difference']['left_y_n']))
            r_x_n = np.nansum(np.array(self.d[pl]['difference']['right_x_n'], dtype=float) / len(self.d[pl]['difference']['right_x_n']))
            r_y_n = np.nansum(np.array(self.d[pl]['difference']['right_y_n'], dtype=float) / len(self.d[pl]['difference']['right_y_n']))
            d1 = {'track': [name +'_track_'+str(pl)]*len(self.d[pl]['frames']), 'frames':self.d[pl]['frames'], 'second': np.array(self.d[pl]['frames']) // 30,
                  'left_x':self.d[pl]['difference']['left_x'], 'left_y':self.d[pl]['difference']['left_y'],
                  'right_x':self.d[pl]['difference']['right_x'], 'right_y':self.d[pl]['difference']['right_y']}
            df1 = pd.DataFrame(data=d1)

            df1['left_x'] = df1['left_x']/ np.sqrt(l_x_n)
            df1['left_y'] = df1['left_y']/ np.sqrt(l_y_n)

            df1['right_x'] = df1['right_x']/ np.sqrt(r_x_n)
            df1['right_y'] = df1['right_y']/ np.sqrt(r_y_n)

            if which == 'draw':
                df1['draw'] = 0
                df_h = df1[(df1['frames'] >= st) & (df1['frames'] <= en)]
                df1.loc[df_h.index, ['draw']] = 1
            elif which == 'begin':
                df1['draw'] = 0
                df_h = df1[(df1['frames'] >= (st+5)) & (df1['frames'] <= (st-5))]
                df1.loc[df_h.index, ['draw']] = 1
            elif which == 'end':
                df1['draw'] = 0
                df_h = df1[(df1['frames'] >= (en+5)) & (df1['frames'] <= (en-5))]
                df1.loc[df_h.index, ['draw']] = 1

            #fill missing value
            data1 = df1.copy()
            data1 = data1.fillna(100)
            for col in data1.columns:
                if col in ['left_x', 'left_y', 'right_x', 'right_y']:
                    for i in range(len(data1[col])):
                        j, n = i, 0
                        while data1[col][j] == 100 and j < len(data1[col]) -1:
                            n += 1
                            j += 1
                            if j == len(data1[col]) - 1 and data1[col][j] == 100:
                                n += 1
                                j +=1
                                break
                        if n > 1:
                            mean_arr = []
                            h = j - n - 1
                            while h > 0 and h >= (j - (n + int(n/2))):
                                if data1[col][h] != 100:
                                    mean_arr.append(data1[col][h])
                                h -= 1
                            h = j 
                            while h < len(data1[col]) -1  and h <= (j + int(n/2)):
                                if data1[col][h] != 100:
                                    mean_arr.append(data1[col][h])
                                h += 1
                            idx = np.arange(i,j)
                            data1.loc[idx, [col]] = np.mean(mean_arr)
                        elif n == 1:
                            if i == 0:
                                data1.loc[i, [col]] = data1[col][i+1]
                            elif i == len(data1[col]) - 1:
                                data1.loc[i, [col]] = data1[col][i-1]
                            else:
                                data1.loc[i, [col]] = np.mean([data1[col][i+1], data1[col][i-1]])
            datas.append(data1)

        final = pd.concat(datas, ignore_index=True)
        return final



    def motion_relative_to_the_frame(self, label: str = 'sec', ext: str = 'png', save: bool = False, close: bool = False):
        self.__checker()

        for pl in self.d.keys():
            if len(self.d[pl]['frames']) != 0:
                c = np.random.choice(['b','g','r','c','m','y','k'])
                name = f'Step movement coordinates for track {pl}'
                if close == False:
                    print(name)
                #fig.suptitle(name, fontsize = 24)
                x = self.d[pl]['frames']
                
                y = ['left_x', 'left_y', 'right_x', 'right_y']
                y_sign = ['Left leg, X position', 'Left leg, Y position', 'Right leg, X position', 'Right leg, Y position']
                
                if label == 'sec':
                    labels_x = list(self.__frames_to_sec(pl=pl))
                    if self.frames_per_s * labels_x[-1] < self.d[pl]['frames'][-1]:
                        labels_x.append(labels_x[-1] + 1)
                
                        
                for i in range(2):
                    for h in range(2):
                        fig, axes = plt.subplots(figsize=(20, 5))
                        plt.plot(x, self.d[pl]['coords'][y[i+2*h]], color=c)
                        #plt.suptitle(f"{y_sign[i+2*h]}", fontsize=15)
                        if close == False:
                            print(y_sign[i+2*h])
                        if i+2*h == 1 or i+2*h == 3:
                            plt.gca().invert_yaxis()
                        plt.ylabel("coordinates", fontsize=10)
                        if label == 'sec':
                            Axis.set_major_locator(axes.xaxis,ticker.MultipleLocator(self.frames_per_s))
                            axes.set_xticklabels(labels_x, fontsize = 10)
                            plt.xlabel("seconds", fontsize=10)
                        else:
                           plt.xlabel("frames", fontsize=10)
                        
                        if save:
                            filename = name + " " + y_sign[i+2*h] +'.' + ext
                            self.__save(filename=filename)
                        if close:
                            plt.close()
                        else:
                            plt.show()
                    
        
                  
    def cyclogram(self, label: str = 'sec', ext: str = 'png', save: bool = False, close: bool = False):
        self.__checker()

        labels_y = ['on ground', 'in air', 'on ground', 'in air']
        
        for pl in self.d.keys():
            c = np.random.choice(['b','g','r','c','m','y','k'])
            fig, axes = plt.subplots( figsize=(15, 3))
            name = f'Cyclogram for track {pl}'
            #fig.suptitle(name, fontsize = 18)
            print(name)
            x = self.d[pl]['frames']
            
            axes.plot(x, self.d[pl]['legs']['right'], color=c)
            axes.plot(x, self.d[pl]['legs']['left'], color=c)
            
            axes.yaxis.set_major_locator(ticker.FixedLocator([1,2,3,4]))
            axes.set_yticklabels(labels_y, fontsize = 10)

            if label == 'sec':
                labels_x = list(self.__frames_to_sec(pl=pl))
                if self.frames_per_s * labels_x[-1] < self.d[pl]['frames'][-1]:
                    labels_x.append(labels_x[-1] + 1)

                axes.xaxis.set_major_locator(ticker.MultipleLocator(self.frames_per_s))
                axes.set_xticklabels(labels_x, fontsize = 10)
                axes.set_xlabel("seconds", fontsize=10)
            else:
                axes.set_xlabel("frames", fontsize=10)

            if save:
                filename = name + '.' + ext
                self.__save(filename=filename)
            
            if close:
                plt.close()
            else:
                plt.show()


    def bird_view_gr(self, label: str = 'sec', ext: str = 'png', save: bool = False, close: bool = False):
        self.__checker()

        for pl in self.d.keys():
            if len(self.d[pl]['frames']) != 0:
                c = np.random.choice(['b','g','r','c','m','y','k'])
                name = f'bird view movement coordinates for track {pl}'
                print(name)
                #fig.suptitle(name, fontsize = 24)
                x = self.d[pl]['frames']
                
                y = ['left_x', 'left_y', 'right_x', 'right_y']
                y_sign = ['Left leg, X position', 'Left leg, Y position', 'Right leg, X position', 'Right leg, Y position']
                
                if label == 'sec':
                    labels_x = list(self.__frames_to_sec(pl=pl))
                    if self.frames_per_s * labels_x[-1] < self.d[pl]['frames'][-1]:
                        labels_x.append(labels_x[-1] + 1)
                
                        
                for i in range(2):
                    for h in range(2):
                        fig, axes = plt.subplots(figsize=(20, 5))
                        if len(self.d[pl]['bird_view'][y[i+2*h]]) < len(x):
                            x = x[:len(self.d[pl]['bird_view'][y[i+2*h]])]
                        plt.plot(x, self.d[pl]['bird_view'][y[i+2*h]], color=c)
                        #plt.suptitle(f"{y_sign[i+2*h]}", fontsize=15)
                        print(y_sign[i+2*h])
                        if i+2*h == 1 or i+2*h == 3:
                            plt.gca().invert_yaxis()
                        plt.ylabel("coordinates", fontsize=10)
                        if label == 'sec':
                            Axis.set_major_locator(axes.xaxis,ticker.MultipleLocator(self.frames_per_s))
                            axes.set_xticklabels(labels_x, fontsize = 10)
                            plt.xlabel("seconds", fontsize=10)
                        else:
                           plt.xlabel("frames", fontsize=10)
                        
                        if save:
                            filename = name + " " + y_sign[i+2*h] +'.' + ext
                            self.__save(filename=filename)
                        if close:
                            plt.close()
                        else:
                            plt.show()


    def movement_viz(self, image_path, H, flag:bool = True):
        cols = [(255,255,0),(0,0,255),(255,0,0),(0,255,255)]
        c, num = 0, 0
        image = cv.imread(image_path, -1)
        image_copy = image.copy()

        for h in range(len(self.d[list(self.d.keys())[0]]['coords']['left_x'])):
            image_copy = image.copy()
            for pl in self.d.keys():
                if h < len(self.d[pl]['coords']['left_x']) or h < len(self.d[pl]['coords']['left_y']):
                    if self.d[pl]['coords']['left_x'][h] != None and self.d[pl]['coords']['left_y'][h] != None:
                        left = np.array([self.d[pl]['coords']['left_x'][h], self.d[pl]['coords']['left_y'][h]]).reshape((1, 2))
                        pts1 = left.reshape(-1,1,2).astype(np.float32)
                        dst1_l = cv.perspectiveTransform(pts1, H)
                        dst1_l = dst1_l.reshape((1,2))
                        self.d[pl]['bird_view']['left_x'].append(dst1_l[0][0])
                        self.d[pl]['bird_view']['left_y'].append(dst1_l[0][1])
                        center_coordinates = (int(dst1_l[:,0][0]), int(dst1_l[:,1][0]))
                        cv.circle(image_copy, center_coordinates, 5, cols[c], -1)
                    else:
                        self.d[pl]['bird_view']['left_x'].append(None)
                        self.d[pl]['bird_view']['left_y'].append(None)


                if h < len(self.d[pl]['coords']['right_x']) or h < len(self.d[pl]['coords']['right_y']):
                    if self.d[pl]['coords']['right_x'][h] != None and self.d[pl]['coords']['right_y'][h] != None:
                        left = np.array([self.d[pl]['coords']['right_x'][h], self.d[pl]['coords']['right_y'][h]]).reshape((1, 2))
                        pts1 = left.reshape(-1,1,2).astype(np.float32)
                        dst1_r = cv.perspectiveTransform(pts1, H)
                        dst1_r = dst1_r.reshape((1,2))
                        self.d[pl]['bird_view']['right_x'].append(dst1_r[0][0])
                        self.d[pl]['bird_view']['right_y'].append(dst1_r[0][1])
                        center_coordinates = (int(dst1_r[:,0][0]), int(dst1_r[:,1][0]))
                        cv.circle(image_copy, center_coordinates, 5, cols[c], -1)
                    else:
                        self.d[pl]['bird_view']['right_x'].append(None)
                        self.d[pl]['bird_view']['right_y'].append(None)

                if (c+1) % 4 == 0:
                    c = 0
                else:
                    c+=1
            if flag:
                cv.imwrite('im.%.3d.jpg'%(num) , image_copy)
            else:
                cv.imwrite('im_s.%.3d.jpg'%(num) , image_copy)
            
            num+=1 
            image_copy = image.copy()


    def __cell_checker(self, x, y):
        cell_num = -1
        if 0 < x <= 237 and 0 < y <= 250:
            return 0
        elif 237 < x <= 474 and 0 < y <= 250:
            return 1
        elif 474 < x <= 712 and 0 < y <= 250:
            return 2
        elif 0 < x <= 237 and 250 < y <= 501:
            return 3
        elif 237 < x <= 474 and 250 < y <= 501:
            return 4
        elif 474 < x <= 712 and 250 < y <= 501:
            return 5
        elif 0 < x <= 237 and 501 < y <= 750:
            return 6
        elif 237 < x <= 474 and 501 < y <= 750:
            return 7
        elif 474 < x <= 712 and 501 < y <= 750:
            return 8
        elif 0 < x <= 237 and 750 < y <= 1002:
            return 9
        elif 237 < x <= 474 and 750 < y <= 1002:
            return 10
        elif 474 < x <= 712 and 750 < y <= 1002:
            return 11
        return cell_num

    def square_stat(self, image_path:str = '', h:np.array = np.array([0])) -> dict:
        c = 0
        cell_d = {}
        for x in range(12):
            cell_d[x] = [0,0,0,0]
        
        for h in range(len(self.d[list(self.d.keys())[0]]['coords']['left_x'])):
            for pl in self.d.keys():
                if h < len(self.d[pl]['coords']['left_x']) or h < len(self.d[pl]['coords']['left_y']):
                    left_x = self.d[pl]['bird_view']['left_x'][h]
                    left_y = self.d[pl]['bird_view']['left_y'][h]
                    if left_x != None and left_y != None:
                        cell_l = self.__cell_checker(left_x, left_y)
                        cell_d[cell_l][c] += 1

                if h < len(self.d[pl]['coords']['right_x']) or h < len(self.d[pl]['coords']['right_y']):
                    right_x = self.d[pl]['bird_view']['right_x'][h]
                    right_y = self.d[pl]['bird_view']['right_y'][h]
                    if right_x != None and right_y != None:
                        cell_r = self.__cell_checker(right_x, right_y)
                        cell_d[cell_r][c] += 1

                if (c+1) % 4 == 0:
                    c = 0
                else:
                    c+=1
        if image_path != '':
            self.movement_viz(image_path=image_path, H=h, flag=False)
        return cell_d