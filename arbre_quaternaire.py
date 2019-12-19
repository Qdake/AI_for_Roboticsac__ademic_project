import copy
import numpy as np
import matplotlib.pyplot as plt

class Quadtree:
    '''alpha maximium depth of the tree
        beta maximum number of points which may fall within a leaf vertex before it is devided
    '''
    def __init__(self,x0,x1,y0,y1,alpha=7,beta=80,profondeur=0):  # alpha=7,beta=80 best parametres given by the article
        self.alpha = alpha
        self.beta = beta
        self.content = list()
        self.nb_individu_stoke_noeud = 0
        self.profondeur = profondeur

        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.tl = None
        self.tr = None
        self.bl = None
        self.br = None

    def ajout(self,individu):
#        print("**avant ajouter {} content {} nb_noeud: {} ".format(individu.bd,[i.bd for i in self.content],self.nb_individu_stoke_sous_arbre))

        if self.nb_individu_stoke_noeud < self.beta:
            self.content.append(individu)
            individu.profondeur = self.profondeur
            self.nb_individu_stoke_noeud += 1
            return True
        else:
            if self.profondeur == self.alpha:
                return False
            if self.tl == None:
                self.tl = Quadtree(self.x0,(self.x0+self.x1)/2,(self.y0+self.y1)/2,self.y1,alpha=self.alpha,beta=self.beta,profondeur=self.profondeur+1)
            if self.tr == None:
                self.tr = Quadtree((self.x0+self.x1)/2,self.x1,(self.y0+self.y1)/2,self.y1,alpha=self.alpha,beta=self.beta,profondeur=self.profondeur+1)
            if self.bl == None:
                self.bl = Quadtree(self.x0,(self.x0+self.x1)/2,self.y0,(self.y0+self.y1)/2,alpha=self.alpha,beta=self.beta,profondeur=self.profondeur+1)
            if self.br == None:
                self.br = Quadtree((self.x0+self.x1)/2,self.x1,self.y0,(self.y0+self.y1)/2,alpha=self.alpha,beta=self.beta,profondeur=self.profondeur+1)
            self.content.append(individu)

            #print("content apres append ",[i.bd for i in self.content])
            #print("********************************************************************\\n")
            for individu in self.content:
                x,y = individu.bd

                if self.bl.x0 <= x and x < self.bl.x1 and self.bl.y0 <= y and y < self.bl.y1:
                    success = self.bl.ajout(individu)

                elif self.br.x0 <= x and x <= self.br.x1 and self.br.y0 <= y and y < self.br.y1:
                    success = self.br.ajout(individu)

                elif self.tl.x0 <= x and x < self.tl.x1 and self.tl.y0 <= y and y <= self.tl.y1:
                    success = self.tl.ajout(individu)

                elif self.tr.x0 <= x and x <= self.tr.x1 and self.tr.y0 <= y and y <= self.tr.y1:
                    success = self.tr.ajout(individu)

            self.content = []
            self.nb_individu_stoke_noeud = 0
            self.beta = 0
            return success
#            print("apres localisation")
#            print("tl : {} nb_noeud {} nb_sa {}".format([i.bd for i in self.tl.content],self.tl.nb_individu_stoke_noeud,self.tl.nb_individu_stoke_sous_arbre))
#            print("tr : {} nb_noeud {} nb_sa {}".format([i.bd for i in self.tr.content],self.tr.nb_individu_stoke_noeud,self.tr.nb_individu_stoke_sous_arbre))
#            print("bl : {} nb_noeud {} nb_sa {}".format([i.bd for i in self.bl.content],self.bl.nb_individu_stoke_noeud,self.bl.nb_individu_stoke_sous_arbre))
#            print("tl : {} nb_noeud {} nb_sa {}".format([i.bd for i in self.br.content],self.br.nb_individu_stoke_noeud,self.br.nb_individu_stoke_sous_arbre))

    def pprint(self):
        for i in self.content:
            print(i.bd)

    def __repr__(self):
        s = str([e.bd for e in self.content]) + "\n"
        tab = ("\t" * self.profondeur)
        if self.tr :
            s += tab + "tr ->" + self.tr.__repr__()
        if self.tl :
            s += tab + "tl ->" + self.tl.__repr__()
        if self.br :
            s += tab + "br ->" + self.br.__repr__()
        if self.bl :
            s += tab + "bl ->" + self.bl.__repr__()
        return s

    def plot(self,imsize=600):
        x0, x1, y0, y1 = self.x0, self.x1, self.y0, self.y1


        if self.tr:  # si self.tr == None alors ce noeud est un feuille , et pas besoin d'etre separer en quatre
            plt.plot([float(x1+x0)/2,float(x1+x0)/2],[y0,y1], 'g-', linewidth=3.0/(self.profondeur+1))
            plt.plot([x0,x1],[float(y1+y0)/2,float(y1+y0)/2], 'g-', linewidth=3.0/(self.profondeur+1))

        if self.tr:
            self.tr.plot()
        if self.tl:
            self.tl.plot()
        if self.br:
            self.br.plot()
        if self.bl:
            self.bl.plot()




### debbug
class Individu:
    def __init__(self,x,y):
        self.bd=(x,y)


#t = [[2,14],[9,15],[11.5,15.5],[10.5,14.5],[11.5,14.5],[12,14.5],[5,4],[6,7],[12.5,2],[14,5],[13,6],[15,9],[14.5,9.1],[13,9],(13,14),[14,15]]
#t = [np.random.uniform(0,16,size=2) for _ in range(300)]

#T = [Individu(i[0],i[1])for i in t]
#b = Quadtree(0,16,0,16,alpha=7,beta=1)

#print(b)

#b.plot()
