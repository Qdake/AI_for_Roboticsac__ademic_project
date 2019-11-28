import copy


class Quadtree:
    '''alpha maximium depth of the tree
        beta maximum number of points which may fall within a leaf vertex before it is devided
    '''
    def __init__(self,x0,x1,y0,y1,type="noeud",alpha_=7,beta_=80):  # alpha=7,beta=80 best parametres given by the article
        self.alpha = alpha_
        self.alpha_user = alpha_
        self.beta_user = beta_
        self.content = list()
        self.nb_individu_stoke_noeud = 0
        self.nb_individu_stoke_sous_arbre = 0

        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.tl = None
        self.tr = None
        self.bl = None
        self.br = None
        if type == "racine":
            self.beta = 0    #nombre maximum d'individu que on peut stoker dans ce noeud
            self.profondeur = 0
        else:
            self.beta = beta_
    def ajout(self,individu):
        print("**avant ajouter {} content {} nb_noeud: {} ".format(individu.bd,[i.bd for i in self.content],self.nb_individu_stoke_sous_arbre))
        if self.nb_individu_stoke_noeud < self.beta:
#            print("self.beta = ",self.beta)
            self.nb_individu_stoke_sous_arbre += 1
            self.content.append(copy.deepcopy(individu))        
            self.nb_individu_stoke_noeud += 1
            return True
        else:
            if self.profondeur == self.alpha:
                return False
            if self.tl == None:
                self.tl = Quadtree(self.x0,(self.x0+self.x1)/2,(self.y0+self.y1)/2,self.y1,alpha_=self.alpha_user,beta_=self.beta_user)
                self.tl.profondeur = self.profondeur+1
            if self.tr == None:
                self.tr = Quadtree((self.x0+self.x1)/2,self.x1,(self.y0+self.y1)/2,self.y1,alpha_=self.alpha_user,beta_=self.beta_user)
                self.tr.profondeur = self.profondeur+1
            if self.bl == None:
                self.bl = Quadtree(self.x0,(self.x0+self.x1)/2,self.y0,(self.y0+self.y1)/2,alpha_=self.alpha_user,beta_=self.beta_user)
                self.bl.profondeur = self.profondeur+1
            if self.br == None:
                self.br = Quadtree((self.x0+self.x1)/2,self.x1,self.y0,(self.y0+self.y1)/2,alpha_=self.alpha_user,beta_=self.beta_user)
                self.br.profondeur = self.profondeur+1
            self.content.append(individu)
            print("content apres append ",[i.bd for i in self.content])
            print("********************************************************************\\n")
            for individu in self.content:
                x,y = individu.bd
                print("***[",x,",",y,"]")
                # bl
                if self.bl.x0 <= x and x < self.bl.x1 and self.bl.y0 <= y and y < self.bl.y1:
                    print("cond bl verifiee")
                    print("self.bl.content avant ",[i.bd for i in self.bl.content])
                    succes = self.bl.ajout(individu)
                    if succes:
                        self.nb_individu_stoke_sous_arbre += 1
                    print("self.bl.content apres ",[i.bd for i in self.bl.content])


                # br
                if self.br.x0 <= x and x <= self.br.x1 and self.br.y0 <= y and y < self.br.y1:
                    print("cond br verifiee")
                    print("self.br.content avant ",[i.bd for i in self.br.content])
                    succes = self.br.ajout(individu)
                    if succes:
                        self.nb_individu_stoke_sous_arbre += 1
                    print("self.br.content apres ",[i.bd for i in self.br.content])
                    

                if self.tl.x0 <= x and x < self.tl.x1 and self.tl.y0 <= y and y <= self.tl.y1:
                    print("cond tl verifiee")
                    print("self.tl.content vant ",[i.bd for i in self.tl.content])
                    succes = self.tl.ajout(individu)
                    if succes:
                        self.nb_individu_stoke_sous_arbre += 1
                    print("self.tl.content apres ",[i.bd for i in self.tl.content])


                if self.tr.x0 <= x and x <= self.tr.x1 and self.tr.y0 <= y and y <= self.tr.y1:
                    print("cond tr verifiee")
                    print("self.tr.content avant ",[i.bd for i in self.tr.content])
                    succes = self.tr.ajout(individu)
                    if succes:
                        self.nb_individu_stoke_sous_arbre += 1
                    print("self.tr.content apres ",[i.bd for i in self.tr.content])
                    
            self.content = []
            self.nb_individu_stoke_noeud = 0
            self.beta = 0
            print("apres localisation")
            print("tl : {} nb_noeud {} nb_sa {}".format([i.bd for i in self.tl.content],self.tl.nb_individu_stoke_noeud,self.tl.nb_individu_stoke_sous_arbre))
            print("tr : {} nb_noeud {} nb_sa {}".format([i.bd for i in self.tr.content],self.tr.nb_individu_stoke_noeud,self.tr.nb_individu_stoke_sous_arbre))
            print("bl : {} nb_noeud {} nb_sa {}".format([i.bd for i in self.bl.content],self.bl.nb_individu_stoke_noeud,self.bl.nb_individu_stoke_sous_arbre))
            print("tl : {} nb_noeud {} nb_sa {}".format([i.bd for i in self.br.content],self.br.nb_individu_stoke_noeud,self.br.nb_individu_stoke_sous_arbre))    
    def pprint(self):
        for i in self.content:
            print(i.bd)
### debbug
class Individu:
    def __init__(self,x,y):
        self.bd=(x,y)
t = [[2,14],[9,15],[11.5,15.5],[10.5,14.5],[11.5,14.5],[5,4],[6,7],[12.5,2],[14,5],[13,6],[15,9],[14.5,9.1],[13,9],(13,14)]
T = [Individu(i[0],i[1])for i in t]
b = Quadtree(0,16,0,16,"racine",1,2)
for i in T:
    b.ajout(i)
