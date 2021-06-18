#integrated 

import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pygame.mixer
import copy
import time
import random
import switchbot_my

H = 4
W = 3
class Animal_Shougi_Super:
    def __init__(self,):
        self.Bint=np.array([[-1,-1,-1],[0,-1,0],[0,1,0],[1,1,1]])#盤面：１がターンの人、-1が相手
        self.Bstr=np.array([["K","L","Z"],["#","H","#"],["#","H","#"],["Z","L","K"]])#盤面、コマ名
        self.m_now=np.array([["#","#"],["#","#"],["#","#"]])#ターンの人の墓地K,Z,H
        self.m_next=np.array([["#","#"],["#","#"],["#","#"]])#墓地
        self.turn=0#ターン数
        self.first=1#先手が誰か
        self.first_change=0#ゲームがスタートしたら1になる
        self.start=0#ゲームスタートしたターン数
        # self.friend=int(random.random()//0.5)+1
        self.friend=2  # sente
        self.movable = {'K': np.array([[0, 1], [0, -1], [1, 0], [-1, 0]]),
           'Z': np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]]),
           'L': np.array([[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]]),
           'H': np.array([[-1, 0]]),
           'N': np.array([[0, 1], [0, -1], [1, 0], [-1, 0], [-1, 1], [-1, -1]])
        }#可動域
        #self.ad="./Downloads/木下vs坂部/"#写真保存するアドレス
        self.ad="./pictures/"
        self.src_pts=[]
        self.src_pts+=self.create_cells([[670,530],[1400,1040]],4,3)#盤面の最低限2すみ
        self.src_pts+=self.create_cells([[640,50],[905,400]],2,3)#墓地１（左）の最低限2すみ
        self.src_pts+=self.create_cells([[1060,30],[1350,390]],2,3)#墓地２（右）の最低限2すみ
        self.src_pts=np.array(self.src_pts, dtype=np.int32)
    def convert(self,x):
        n={"H":len(np.argwhere(x=="H")),"K":len(np.argwhere(x=="K")),"Z":len(np.argwhere(x=="Z"))}
        return n

    def move(self,ps, pi, n1, n2, i, j, ni, nj):
        catch = False
        trying = False
        ps_ = ps.copy()
        pi_ = pi.copy()
        n1_ = n1.copy()
        n2_ = n2.copy()
        if (ni == 0 and ps_[i][j] == 'L'):
            trying = True
        if (ni == 0 and ps_[i][j] == 'H'):
            ps_[i][j] = 'N'
        if (pi_[ni][nj] == -1):
            if (ps_[ni][nj] == 'L'):
                catch = True
            elif (ps_[ni][nj] == 'N'):
                n1_['H'] += 1
            else:
                n1_[ps_[ni][nj]] += 1
        ps_[ni][nj] = ps_[i][j]
        ps_[i][j] = '#'
        pi_[ni][nj] = 1
        pi_[i][j] = 0
        return ps_, pi_, n1_, n2_, catch, trying


    #持ち駒cを(ni,nj)に打つ
    def mochigoma(self,ps, pi, c, n1, n2, ni, nj):
        ps_ = ps.copy()
        pi_ = pi.copy()
        n1_ = n1.copy()
        n2_ = n2.copy()
        n1_[c] -= 1
        ps_[ni][nj] = c
        pi_[ni][nj] = 1
        return ps_, pi_, n1_, n2_


    def isrange(self,x, y):
        if (0 <= x and x <= H - 1 and 0 <= y and y <= W - 1):
            return True
        else:
            return False


    def swap(self,ps, pi, n1, n2):
        pi = (-1) * pi
        ps = np.flip(ps)
        pi = np.flip(pi)
        return ps, pi, n2, n1


    def check(self,ps, pi):
        for i in range(H):
            for j in range(W):
                if pi[i][j] == 1:
                    for k in self.movable[ps[i][j]]:
                        if self.isrange(i + k[0], j + k[1]) and pi[i + k[0]][j + k[1]] == -1 and ps[i + k[0]][j + k[1]] == 'L':
                            return True
        return False


    #nは奇数、n手以内に勝つか判定
    def win(self,ps, pi, n1, n2, n):
        if n == 0:
            return True
        for i in range(H):
            for j in range(W):
                if pi[i][j] == 1:
                    for k in self.movable[ps[i][j]]:
                        if self.isrange(i + k[0], j + k[1]) and pi[i + k[0]][j + k[1]] != 1:
                            nps, npi, nn1, nn2, catch, trying = self.move(ps, pi, n1, n2, i, j, i + k[0], j + k[1])
                            if catch:
                                return True
                            if trying and not self.check(np.flip(nps), (-1) * np.flip(npi)):
                                return True
                            if not self.win(np.flip(nps), (-1) * np.flip(npi), nn2, nn1, n-1):
                                return True
                if pi[i][j] == 0:
                    for key in n1:
                        if n1[key] >= 1:
                            nps, npi, nn1, nn2 = self.mochigoma(ps, pi, key, n1, n2, i, j)
                            if not self.win(np.flip(nps), (-1) * np.flip(npi), nn2, nn1, n-1):
                                return True
        return False

    def create_cells(self,a,hori,verti):#格子点を作成、aが最低限2隅、horiが横のマス数、vertiが縦のマス数
        c=[]
        dx=(a[1][0]-a[0][0])//hori
        dy=(a[1][1]-a[0][1])//verti
        for i in range (0,hori+1):
            for j in range (0,verti+1):
                c.append([a[0][0]+dx*i,a[0][1]+dy*j])
        #print(len(c))
        return c

    def piece_position(self,file):
        im=cv2.imread(self.ad+file,cv2.IMREAD_GRAYSCALE)
        im_col=cv2.imread(self.ad+file)
        #im_marked = im.copy()
        src_pts=[]
        it=1
        kernel_gradient_3x3=np.array([
                                [ -1,  0,  1],
                                [ -2,  0,  2],
                                [-1, 0, 1]
                                ], np.float32)
        A=np.zeros(24)
        for i in range (0,39):
            if i in [3,7,11,15,16,17,18,19,23,27,28,29,30,31,35]:
                continue
            image=im[self.src_pts[i][1]:self.src_pts[i+1][1],self.src_pts[i][0]:self.src_pts[i+5][0]]#.copy()
            image_col=im_col[self.src_pts[i][1]:self.src_pts[i+1][1],self.src_pts[i][0]:self.src_pts[i+5][0],:]
            cv2.imwrite(self.ad+file+"_square"+str(it)+".jpg", image_col)
            img_gradient_3x3=cv2.filter2D(image, -1, kernel_gradient_3x3)
            n,m=img_gradient_3x3.shape
            s=0
            for i in range(10//2,(n-10)//2):
                for j in range (10,m-10):
                    #if img_gradient_3x3[i][j]>0:
                    s+=img_gradient_3x3[2*i][j]
            if it>=13:
                if (it % 3 == 0):
                    A[it - 1] = s * 1.3
                else:
                    A[it-1]=s*1.2
            else:
                A[it-1]=s
            print(it, A[it - 1])
            #cv2.imwrite(self.ad+file+"_square"+str(it)+"_grad.jpg", img_gradient_3x3)
            it+=1
        index=A.argsort()[-8:][::-1]
        for i in range (0,24):
            if i not in index:
                A[i]=0
            else:
                A[i]=3
        B=np.fliplr(np.transpose(A[0:12].reshape((3,4),order="F")))#盤面
        m1=A[12:18].reshape((3,2),order="F")#左墓地
        m2=A[18:25].reshape((3,2),order="F")#右墓地

        return B,m1,m2

    def color_likelihood(self,im1,im2):#二つの画像の色分布の距離を計算
        img1=cv2.imread(self.ad+im1)
        img2=cv2.imread(self.ad+im2)
        n,m,z=img1.shape
        s=0
        for i in range (0,3):#RGB全てに対してやる
            hist1=cv2.calcHist([img1[10:n-10,10:m-10,:]],[i],None,[256],[0,256])
            hist2=cv2.calcHist([img2[10:n-10,10:m-10,:]],[i],None,[256],[0,256])
            j=np.argmax(hist1)
            k=np.argmax(hist2)
            #print(j-k)
            if abs(j-k)>19: return -100000000
            #ノイズが必ず入るのでピークは合わせる
            if j>k:
                #diff=hist1[j-k:]-hist2[:256-(j-k)]
                #s+=np.linalg.norm(diff[j-k:256-(j-k)])
                s+=cv2.compareHist(hist1[j-k:],hist2[:256-(j-k)],cv2.HISTCMP_CORREL)
            else:
                #diff=hist1[k-j:]-hist2[:256-(k-j)]
                #s+=np.linalg.norm(diff[k-j:256-(k-j)])
                s+=cv2.compareHist(hist1[k-j:],hist2[:256-(k-j)],cv2.HISTCMP_CORREL)
        return s/3
    
    def update(self,Bint,m1,m2):#更新則
        #まず開始時点を定める、最初はself.first_change=0である。

        if self.first_change==1: #ゲーム開始後
            if self.turn%2==(self.start+self.first)%2:#左の人のターンのとき
                
                B_before=self.Bint.copy()
                B_after=np.flipud(np.fliplr(Bint.copy()))#必ずしたをそのターンの人に向けるフリップする(左の時だけ特別な処置) 
                B_change=B_before-B_after#盤面変化
                A_a=m2#ターンの人の墓地
                
            else:
                A_a=m1
                B_before=self.Bint.copy()
                B_after=Bint.copy()
                B_change=B_before-B_after

        elif self.first_change==0:#最初の変化が起きるまではこっち
            
            B_before=self.Bint.copy()
            B_after=Bint.copy()
            B_change=B_before-B_after
            
            if (-1 in B_change):#左が先手なら初めての変化は必ずこっち
                
                self.first=2#先手設定
                self.first_change=1#ゲーム開始
                self.start=self.turn#ゲーム開始ターン数
                B_after=np.flipud(np.fliplr(Bint.copy()))#必ずしたをそのターンの人に向けるフリップする
                B_change=B_before-B_after#盤面変化
                A_a=m2#そのターンの人の墓地
                #print(B_change)

            elif (1 in B_change):#右が先手なら初めての変化は必ずこっち
                A_a=m1
                self.first=1
                self.first_change=1
                self.start=self.turn
                B_after=Bint.copy()
                B_change=B_before-B_after
            
        #以下変化パターンで分ける
        
        if (-3 in B_change) and (1 in B_change):#コマがただ移動した時
            
            #print("in1")
            #B_change で-3が移動先、1が移動元
            
            #インデックスの特定
            i_a=np.where(B_change==-3)[0][0]
            j_a=np.where(B_change==-3)[1][0]
            i_b=np.where(B_change==1)[0][0]
            j_b=np.where(B_change==1)[1][0]
            
            #後は入れ替え
            self.Bint[i_a][j_a]=self.Bint[i_b][j_b].copy()
            self.Bstr[i_a][j_a]=self.Bstr[i_b][j_b].copy()
            
            self.Bint[i_b][j_b]=0
            self.Bstr[i_b][j_b]="#"
            
            if i_a==0 and self.Bstr[i_a][j_a]=="H":
                
                self.Bstr[i_a][j_a]="N"
            
        elif (1 in B_change):#相手のコマをとるムーブの時（一番厄介）
            #print("in2")
            
            
            A_b=self.m_now #1ターン前の墓地の状況
            
            A_change=np.logical_xor((A_b!="#"),(A_a!=0))#墓地の変化
            #墓地の変化した場所を特定、それによってとられたコマの種類がわかる
            i=np.where(A_change==True)[0][0]
            j=np.where(A_change==True)[1][0]
            taken=np.array([["K","K"],["Z","Z"],["H","H"]])[i][j]
            
            #移動したコマを特定
            i_b=np.where(B_change==1)[0][0]
            j_b=np.where(B_change==1)[1][0]
            moved=self.Bstr[i_b][j_b]
            
            #取られたコマを参照して、移動したコマの移動先を特定する。
            #すなわち、移動したコマの可動域を見て取られたコマと同じ種類の相手のコマがあるか探索する（最大2、最小１）
            #possibilityに候補を格納
            
            move_to=self.movable[moved]
            possibility=[]
            #print(move_to,i_b,j_b)
            
            for k in move_to:
                
                if 0<=i_b+k[0]<=3 and 0<=j_b+k[1]<=2:
                    #print(self.Bint[i_b+k[0]][j_b+k[1]],self.Bstr[i_b+k[0]][j_b+k[1]])
                    #print(taken)
                    if taken=="H":
                        if self.Bint[i_b+k[0]][j_b+k[1]]==-1 and (self.Bstr[i_b+k[0]][j_b+k[1]]==taken or self.Bstr[i_b+k[0]][j_b+k[1]]=="N"):
                            possibility.append([i_b+k[0],j_b+k[1]])
                    else:
                        if self.Bint[i_b+k[0]][j_b+k[1]]==-1 and self.Bstr[i_b+k[0]][j_b+k[1]]==taken:
                            possibility.append([i_b+k[0],j_b+k[1]])
            
            #print(possibility)
            
            if len(possibility)==1:#行先が1通りしかない場合は、簡単

                self.Bint[i_b][j_b]=0
                self.Bint[possibility[0][0],possibility[0][1]]=1
                self.Bstr[possibility[0][0],possibility[0][1]]=self.Bstr[i_b][j_b].copy()
                self.Bstr[i_b][j_b]="#"

                self.m_now[i][j]=taken
            
                if possibility[0][0]==0 and self.Bstr[possibility[0][0],possibility[0][1]]=="H":
                    
                    self.Bstr[possibility[0][0],possibility[0][1]]="N"


            else:#行先が２通りの場合は色分布の比較が必要
                
                if self.turn%2==(self.start+self.first)%2:#左のターンの場合
                    
                    p=[10,7,4,1,11,8,5,2,12,9,6,3]
                    #print(possibility[0][0]+possibility[0][1]*4)
                    #print(possibility[1][0]+possibility[1][1]*4)
                    index_im1=p[possibility[0][0]+possibility[0][1]*4]
                    index_im2=p[possibility[1][0]+possibility[1][1]*4]
                    index_model_im=p[i_b+j_b*4]
                    im1="image"+str(self.turn)+".png_square"+str(index_im1)+".jpg"#候補1の変化後の写真
                    im2="image"+str(self.turn)+".png_square"+str(index_im2)+".jpg"#候補2の変化後の写真
                    im1_before="image"+str(self.turn-1)+".png_square"+str(index_im1)+".jpg"
                    im2_before="image"+str(self.turn-1)+".png_square"+str(index_im2)+".jpg"
                
                    
                    #色分布の距離を計算
                    like1=self.color_likelihood(im1,im1_before)
                    like2=self.color_likelihood(im2,im2_before)
                    
                    if (like1<like2):
                        
                        self.Bint[i_b][j_b]=0
                        self.Bint[possibility[0][0],possibility[0][1]]=1
                        self.Bstr[possibility[0][0],possibility[0][1]]=self.Bstr[i_b][j_b].copy()
                        self.Bstr[i_b][j_b]="#"
                        self.m_now[i][j]=taken
                        
                        if possibility[0][0]==0 and self.Bstr[possibility[0][0],possibility[0][1]]=="H":
                    
                            self.Bstr[possibility[0][0],possibility[0][1]]="N"


                    else:
                        self.Bint[i_b][j_b]=0
                        self.Bint[possibility[1][0],possibility[1][1]]=1
                        self.Bstr[possibility[1][0],possibility[1][1]]=self.Bstr[i_b][j_b].copy()
                        self.Bstr[i_b][j_b]="#"
                        self.m_now[i][j]=taken

                        if possibility[1][0]==0 and self.Bstr[possibility[1][0],possibility[1][1]]=="H":

                            self.Bstr[possibility[1][0],possibility[1][1]]="N"

                        
                        
                else:#右のターンのとき
                    
                    p=[3,6,9,12,2,5,8,11,1,4,7,10]
                    #print(possibility)
                    #print(possibility[0][0]+possibility[0][1]*4)
                    #print(possibility[1][0]+possibility[1][1]*4)
                    index_im1=p[possibility[0][0]+possibility[0][1]*4]
                    index_im2=p[possibility[1][0]+possibility[1][1]*4]
                    index_model_im=p[i_b+j_b*4]
                    im1="image"+str(self.turn)+".png_square"+str(index_im1)+".jpg"
                    im2="image"+str(self.turn)+".png_square"+str(index_im2)+".jpg"
                    im1_before="image"+str(self.turn-1)+".png_square"+str(index_im1)+".jpg"
                    im2_before="image"+str(self.turn-1)+".png_square"+str(index_im2)+".jpg"
                    
                    like1=self.color_likelihood(im1,im1_before)
                    like2=self.color_likelihood(im2,im2_before)
                    
                    if (like1<like2):
                        
                        self.Bint[i_b][j_b]=0
                        self.Bint[possibility[0][0],possibility[0][1]]=1
                        self.Bstr[possibility[0][0],possibility[0][1]]=self.Bstr[i_b][j_b].copy()
                        self.Bstr[i_b][j_b]="#"
                        self.m_now[i][j]=taken
                        
                        if possibility[0][0]==0 and self.Bstr[possibility[0][0],possibility[0][1]]=="H":
                    
                            self.Bstr[possibility[0][0],possibility[0][1]]="N"


                    else:
                        self.Bint[i_b][j_b]=0
                        self.Bint[possibility[1][0],possibility[1][1]]=1
                        self.Bstr[possibility[1][0],possibility[1][1]]=self.Bstr[i_b][j_b].copy()
                        self.Bstr[i_b][j_b]="#"
                        self.m_now[i][j]=taken
                        
                        if possibility[1][0]==0 and self.Bstr[possibility[1][0],possibility[1][1]]=="H":
                    
                            self.Bstr[possibility[1][0],possibility[1][1]]="N"


            #print("under construction")
            
            
            
        elif (-3 in B_change):#コマが墓地から召喚された時
            #print("in3")
            
            A_b=self.m_now
            A_change=np.logical_xor((A_b!="#"),(A_a!=0))#召喚されたコマを特定
                
            i_a=np.where(B_change==-3)[0][0]
            j_a=np.where(B_change==-3)[1][0]
            i_b=np.where(A_change==True)[0][0]
            j_b=np.where(A_change==True)[1][0]
            
            self.Bint[i_a][j_a]=1
            
            self.Bstr[i_a][j_a]=self.m_now[i_b][j_b].copy()
            self.m_now[i_b][j_b]="#"
            
        #print(self.Bint,self.Bstr)
        
        self.Bint=np.flipud(np.fliplr(-self.Bint))
        self.Bstr=np.flipud(np.fliplr(self.Bstr))
        tmp=self.m_now.copy()
        self.m_now=self.m_next.copy()
        self.m_next=tmp.copy()
    
    def step(self):#ターンの経過毎に発動
    
        self.turn+=1 #ターン数+1
        
        name = "pictures/image" + str(self.turn) + ".png"
        cv2.imwrite(name, frame) # ファイル保存
        print("\n")
        
        ##!save picture to "image{turn}.png" e.g. image11.jpg!##
        
        image_name="image"+str(self.turn)+".png"#写真の名前
        Bint,m2,m1=self.piece_position(image_name)#ありなし判断
        
        #print(self.turn)
        #print(m2)
        #print(m1)
        #print(Bint)
        
        #img = mpimg.imread(self.ad+"image"+str(self.turn)+".png")
        #imgplot = plt.imshow(img)
        
        self.update(Bint,m1,m2)#盤面情報更新
        
        #plt.show()
        
deviceid = 2
cap = cv2.VideoCapture(deviceid)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
max_time = 30 #持ち時間
teban = 0 #手番
count = 0 #手数

pygame.mixer.init() #初期化

ass=Animal_Shougi_Super()

while True:

    if teban == 0:
        print("先手の番です。\n")
    else:
        print("後手の番です。\n")
    for t in range(max_time):
        print('残り{}秒'.format(max_time - t))
        if t == 10:
            pygame.mixer.music.load("10sec.ogg")  # 読み込み
            pygame.mixer.music.play(1)
        if t == 20:
            pygame.mixer.music.load("20sec.ogg")  # 読み込み
            pygame.mixer.music.play(1)
        ret, frame = cap.read()

        frame = cv2.line(frame, (630, 50), (650, 50), (0, 0, 255), 2)
        frame = cv2.line(frame, (640, 40), (640, 60), (0, 0, 255), 2)

        frame = cv2.line(frame, (620, 410), (640, 410), (0, 0, 255), 2)
        frame = cv2.line(frame, (630, 400), (630, 420), (0, 0, 255), 2)

        frame = cv2.line(frame, (895, 400), (915, 400), (0, 0, 255), 2)
        frame = cv2.line(frame, (905, 390), (905, 410), (0, 0, 255), 2)

        frame = cv2.line(frame, (885, 40), (905, 40), (0, 0, 255), 2)
        frame = cv2.line(frame, (895, 30), (895, 50), (0, 0, 255), 2)

        frame = cv2.line(frame, (1050, 30), (1070, 30), (0, 0, 255), 2)
        frame = cv2.line(frame, (1060, 20), (1060, 40), (0, 0, 255), 2)

        frame = cv2.line(frame, (1295, 30), (1315, 30), (0, 0, 255), 2)
        frame = cv2.line(frame, (1305, 20), (1305, 40), (0, 0, 255), 2)

        frame = cv2.line(frame, (1070, 390), (1090, 390), (0, 0, 255), 2)
        frame = cv2.line(frame, (1080, 380), (1080, 400), (0, 0, 255), 2)

        frame = cv2.line(frame, (1340, 390), (1360, 390), (0, 0, 255), 2)
        frame = cv2.line(frame, (1350, 380), (1350, 400), (0, 0, 255), 2)

        frame = cv2.line(frame, (660, 530), (680, 530), (0, 0, 255), 2)
        frame = cv2.line(frame, (670, 520), (670, 540), (0, 0, 255), 2)

        frame = cv2.line(frame, (1333, 510), (1353, 510), (0, 0, 255), 2)
        frame = cv2.line(frame, (1343, 500), (1343, 520), (0, 0, 255), 2)

        frame = cv2.line(frame, (648, 1055), (668, 1055), (0, 0, 255), 2)
        frame = cv2.line(frame, (658, 1045), (658, 1065), (0, 0, 255), 2)

        frame = cv2.line(frame, (1390, 1040), (1410, 1040), (0, 0, 255), 2)
        frame = cv2.line(frame, (1400, 1030), (1400, 1050), (0, 0, 255), 2)

        cv2.imshow("camera", frame)
        # キー入力を待つ
        k = cv2.waitKey(1000) & 0xff
        if k == 13:
            # Enterキーで画像を保存
            ass.step()
            print (ass.Bstr)
            print (ass.Bint)
            if(ass.turn%2==(ass.start+ass.friend)%2 and ass.win(ass.Bstr, ass.Bint, ass.convert(ass.m_now), ass.convert(ass.m_next), 3)):
                switchbot_my.fall()
                print ("friend: " + ass.friend)  # 1: gote 2: sente
            teban = 1 - teban
            pygame.mixer.music.stop()
            break
        elif k == ord('q'):
            # 「q」キーが押されたら終了する
            exit()
