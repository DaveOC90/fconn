## Taken from https://stackoverflow.com/questions/45863939/adding-a-color-bar-to-a-chord-diagram-in-python-plotly  ##
## Ideograms: https://plotly.com/python/v3/filled-chord-diagram/ ##
import numpy as np
import plotly
import matplotlib.pyplot as plt
import random
import pdb
import pandas as pd
from collections import Counter
import subprocess

color_range = plt.get_cmap('OrRd')


def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def get_idx_interv(d, D):
    k = 0
    while d > D[k]:
        k += 1
    return k - 1


def deCasteljau(b, t):
    n = len(b)
    a = np.copy(b)  # shallow copy of the list of control points
    for r in range(1, n):
        a[:n - r, :] = (1 - t) * a[:n - r, :] + t * a[1:n - r + 1, :]
    return a[0, :]


def bezierCv(b, nr=5):
    t = np.linspace(0, 1, nr)
    return np.array([deCasteljau(b, t[k]) for k in range(nr)])


def create_circle_coords(r,n):
    spacing=np.linspace(0,2*np.pi,n+1)
    x=r*np.cos(spacing)
    y=r*np.sin(spacing)
    return x,y

def create_circle_coords_rot(r,n,rot=0,start=0,end=2*np.pi):
    # rot is a rotation in terms of pi
    # i.e. 0.5 is a quarter turn is 90 deg
    rot = rot*np.pi
    spacing=np.linspace(start+rot,end+rot,n+1)
    x=r*np.cos(spacing)
    y=r*np.sin(spacing)
    return x,y

def moduloAB(x, a, b): #maps a real number onto the unit circle identified with 
                       #the interval [a,b), b-a=2*PI
        if a>=b:
            raise ValueError('Incorrect interval ends')
        y=(x-a)%(b-a)
        return y+b if y<0 else y+a

def test_2PI(x):
    return 0<= x <2*np.pi


def get_ideogram_ends(ideogram_len, gap):
    ideo_ends=[]
    left=0
    for k in range(len(ideogram_len)):
        right=left+ideogram_len[k]
        ideo_ends.append([left, right])
        left=right+gap
    return ideo_ends

def get_ideogram_ends_diffhem(ideogram_len, gap,start = 0):
    ideo_ends=[]
    left=start

    numEnds = len(ideogram_len)

    gapInds = [round((numEnds/2)-1),numEnds]  

    for k in range(numEnds):
        if k in gapInds:
            right=left+ideogram_len[k]
            ideo_ends.append([left, right])
            left=right+gap
        else:
            right=left+ideogram_len[k]
            ideo_ends.append([left, right])
            left=right
    return ideo_ends

def make_ideogram_arc(R, phi, a=50,rot=0):
    # R is the circle radius
    # phi is the list of ends angle coordinates of an arc
    # a is a parameter that controls the number of points to be evaluated on an arc
    # rot is rotation to make as a fraction of pi

    rot = rot*np.pi
    if not test_2PI(phi[0]) or not test_2PI(phi[1]):
        phi=[moduloAB(t, 0+rot, (2*np.pi)+rot) for t in phi]
    length=(phi[1]-phi[0])% 2*np.pi
    nr=5 if length<=np.pi/4 else int(a*length/np.pi)

    if phi[0] < phi[1]:
        theta=np.linspace(phi[0], phi[1], nr)
    else:
        phi=[moduloAB(t, -np.pi, np.pi) for t in phi]
        theta=np.linspace(phi[0], phi[1], nr)
    return R*np.exp(1j*theta)


def make_ideo_shape(path, line_color, fill_color):
    #line_color is the color of the shape boundary
    #fill_collor is the color assigned to an ideogram
    return  dict(
                  line=dict(
                  color=line_color,
                  width=0.45
                 ),

            path=  path,
            type='path',
            fillcolor=fill_color,
            layer='below'
        )



def run_main(edgedf_stack,netmap,opname):
    print("Prepping Input")

    # What are the labels and the connections
    labels = edgedf_stack.source.unique()
    E = edgedf_stack[['sourceid','destid']].values
    nodesortlist='node'+netmap.oldroi.astype(str).str.zfill(3)

    print('Create circle coords')
    Xn,Yn = create_circle_coords(1,len(labels))
    layt=list(zip(Xn,Yn))

    Weights = edgedf_stack.weight.values


    print("Set constants")
    Dist = [0, dist([1, 0], 2 * [np.sqrt(2) / 2]), np.sqrt(2),
            dist([1, 0], [-np.sqrt(2) / 2, np.sqrt(2) / 2]), 2.0]
    params = [1.2, 1.5, 1.8, 2.1]

    edge_colors = ['#d4daff', '#84a9dd', '#5588c8', '#6d8acf']

    L = len(labels)


    lines = list()
    edge_info = list()
    my_weights = Weights



    print('Create Lines')
    for j, e in enumerate(E):
        if my_weights[j] > 0:

            A = np.array(layt[e[0]-1]) #source node
            B = np.array(layt[e[1]-1]) # destination node
            d = dist(A, B) # distance between them
            K = get_idx_interv(d, Dist) # find which quadrant?
            b = [A, A / params[K], B / params[K], B]
            color = edge_colors[K]
            pts = bezierCv(b) # control points in arc?
            text = nodesortlist[e[0]-1] + ' to ' + nodesortlist[e[1]-1] #+ ' ' + str(Weights[j]) + ' pts'
            mark = deCasteljau(b, 0.9)

            edge_info.append(plotly.graph_objs.Scatter(x=np.array(mark[0]),y=np.array(mark[1]),mode='markers',marker=plotly.graph_objs.scatter.Marker(size=0.5, color=edge_colors),text=text,hoverinfo='text',showlegend=False)) 

            lines.append(plotly.graph_objs.Scatter(x=pts[:, 0],y=pts[:, 1],mode='lines',line=plotly.graph_objs.Line(color='rgba({}, {}, {}, {})'.format(*color_range(my_weights[j])),shape='spline',width=Weights[j]*2),hoverinfo='none',showlegend=False))
        else:
            pass


    ###################################################################




    print('Do outside arcs')



    ideo_colors=['rgba(166,206,227, 1)',
        'rgba(31,120,180, 1)',
        'rgba(178,223,138, 1)',
        'rgba(51,160,44, 1)',
        'rgba(251,154,153, 1)',
        'rgba(227,26,28, 1)',
        'rgba(253,191,111, 1)',
        'rgba(255,127,0, 1)',
        'rgba(202,178,214, 1)',
        'rgba(106,61,154, 1)',
        'rgba(255,255,153, 1)',
        'rgba(177,89,40, 1)']


    L = 10

    row_sum=list(Counter(netmap['label']).values())


    #set the gap between two consecutive ideograms
    gap=0#2*np.pi*0.005
    ideogram_length=2*np.pi*np.asarray(row_sum)/sum(row_sum)-gap*np.ones(L)

    ideo_ends=get_ideogram_ends(ideogram_length, gap)

    z=make_ideogram_arc(1.3, [11*np.pi/6, np.pi/17])

    labels=list(Counter(netmap['label']).keys()) 


    #pdb.set_trace()

    ideograms=[]
    shape_append=[]
    for k in range(len(ideo_ends)):
        z= make_ideogram_arc(1.0, ideo_ends[k])
        zi=make_ideogram_arc(1.1, ideo_ends[k])
        m=len(z)
        n=len(zi)
        ideograms.append(plotly.graph_objs.Scatter(x=z.real,
                                 y=z.imag,
                                 mode='lines',
                                 line=dict(color=ideo_colors[k], shape='spline', width=10),
                                 text=labels[k]+'<br>'+'{:d}'.format(row_sum[k]),
                                 hoverinfo='text',
                                 name=labels[k],
                                 showlegend=True
                                 )
                         )


        path='M '
        for s in range(m):
            path+=str(z.real[s])+', '+str(z.imag[s])+' L '

        Zi=np.array(zi.tolist()[::-1])

        for s in range(m):
            path+=str(Zi.real[s])+', '+str(Zi.imag[s])+' L '
        path+=str(z.real[0])+' ,'+str(z.imag[0])

        
        #layout['shapes'].append(make_ideo_shape(path,'rgb(150,150,150)' , ideo_colors[k]))
        shape_append.append(make_ideo_shape(path,'rgb(150,150,150)' , ideo_colors[k]))



    ##############################################################



    print('Make axis and layout')
    axis = dict(showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title=''
                )

    layout = plotly.graph_objs.Layout(showlegend=True,
                                      autosize=False,
                                      width=800,
                                      height=850,
                                      paper_bgcolor='rgba(255,255,255, 1)',
                                      plot_bgcolor='rgba(255,255,255, 1)',
                                      xaxis=plotly.graph_objs.XAxis(axis),
                                      yaxis=plotly.graph_objs.YAxis(axis),
                                      margin=plotly.graph_objs.Margin(l=40,
                                                                      r=40,
                                                                      b=85,
                                                                      t=100,
                                                                      ),
                                      hovermode='closest',
                                      legend=dict(x=-.1, y=1.2))

    layout['shapes'] = shape_append



    color_trace = plotly.graph_objs.Scatter(x=[0 for _ in my_weights],y=[0 for _ in my_weights],mode='markers',marker=plotly.graph_objs.scatter.Marker(colorscale=[[c / 100.0, 'rgba({}, {}, {}, {})'.format(*color_range(c / 100.0))] for c in range(101)],size=1,color=my_weights,showscale=True),showlegend=False)

    #data = plotly.graph_objs.Data([color_trace] + lines + edge_info + [trace2])
    print('Make data object')
    data = plotly.graph_objs.Data(lines+edge_info+ideograms+[color_trace])
    print('Make fig object')
    fig = plotly.graph_objs.Figure(data=data, layout=layout)


    print('Write image')
    fig.write_image(opname)



def run_main_diffhem(edgedf_stack,netmap,opname, pltX = 800, pltY = 850, legend = True):
    print("Prepping Input")



    # What are the labels and the connections
    labels = edgedf_stack.source.unique()
    E = edgedf_stack[['sourceid','destid']].values
    nodesortlist='node'+netmap.oldroi.astype(str).str.zfill(3)

    print('Create circle coords')

    #set the gap between two consecutive ideograms
    gap=2*np.pi*0.0025


    #Xn,Yn = create_circle_coords_rot(1,len(labels),rot=0.5,start = gap/2,end=(np.pi*2)+gap/2)
    Xn,Yn = create_circle_coords_rot(1,len(labels),rot=0.5)
    layt=list(zip(Xn,Yn))

    Weights = edgedf_stack.weight.values

    #pdb.set_trace()


    print("Set constants")
    Dist = [0, dist([1, 0], 2 * [np.sqrt(2) / 2]), np.sqrt(2),
            dist([1, 0], [-np.sqrt(2) / 2, np.sqrt(2) / 2]), 2.0]
    params = [1.2, 1.5, 1.8, 2.1]

    edge_colors = ['#d4daff', '#84a9dd', '#5588c8', '#6d8acf']

    L = len(labels)


    lines = list()
    edge_info = list()
    my_weights = Weights



    print('Create Lines')
    for j, e in enumerate(E):
        if my_weights[j] > 0:

            A = np.array(layt[e[0]-1]) #source node
            B = np.array(layt[e[1]-1]) # destination node
            d = dist(A, B) # distance between them
            K = get_idx_interv(d, Dist) # find which quadrant?
            b = [A, A / params[K], B / params[K], B]
            color = edge_colors[K]
            pts = bezierCv(b) # control points in arc?
            text = nodesortlist[e[0]-1] + ' to ' + nodesortlist[e[1]-1] #+ ' ' + str(Weights[j]) + ' pts'
            mark = deCasteljau(b, 0.9)

            edge_info.append(plotly.graph_objs.Scatter(x=np.array(mark[0]),y=np.array(mark[1]),mode='markers',marker=plotly.graph_objs.scatter.Marker(size=0.5, color=edge_colors),text=text,hoverinfo='text',showlegend=False)) 

            lines.append(plotly.graph_objs.Scatter(x=pts[:, 0],y=pts[:, 1],mode='lines',line=plotly.graph_objs.Line(color='rgba({}, {}, {}, {})'.format(*color_range(my_weights[j])),shape='spline',width=Weights[j]*2),hoverinfo='none',showlegend=False))
        else:
            pass


    ###################################################################




    print('Do outside arcs')

    #pdb.set_trace()

    '''
    ideo_colors=['rgba(166,206,227, 1)',
        'rgba(31,120,180, 1)',
        'rgba(178,223,138, 1)',
        'rgba(51,160,44, 1)',
        'rgba(251,154,153, 1)',
        'rgba(227,26,28, 1)',
        'rgba(253,191,111, 1)',
        'rgba(255,127,0, 1)',
        'rgba(202,178,214, 1)',
        'rgba(106,61,154, 1)',
        'rgba(255,255,153, 1)',
        'rgba(177,89,40, 1)',
    'rgba(177,89,40, 1)',
        'rgba(255,255,153, 1)',
        'rgba(106,61,154, 1)',
        'rgba(202,178,214, 1)',
        'rgba(255,127,0, 1)',
        'rgba(253,191,111, 1)',
        'rgba(227,26,28, 1)',
        'rgba(251,154,153, 1)',
        'rgba(51,160,44, 1)',
        'rgba(178,223,138, 1)',
        'rgba(31,120,180, 1)',
        'rgba(166,206,227, 1)']
    '''


    ideo_colors=['rgba(166,206,227, 1)',
        'rgba(31,120,180, 1)',
        'rgba(178,223,138, 1)',
        'rgba(51,160,44, 1)',
        'rgba(251,154,153, 1)',
        'rgba(227,26,28, 1)',
        'rgba(253,191,111, 1)',
        'rgba(255,127,0, 1)',
        'rgba(202,178,214, 1)',
        'rgba(106,61,154, 1)',
    'rgba(106,61,154, 1)',
        'rgba(202,178,214, 1)',
        'rgba(255,127,0, 1)',
        'rgba(253,191,111, 1)',
        'rgba(227,26,28, 1)',
        'rgba(251,154,153, 1)',
        'rgba(51,160,44, 1)',
        'rgba(178,223,138, 1)',
        'rgba(31,120,180, 1)',
        'rgba(166,206,227, 1)']

    L = 20

    row_sum=list(Counter(netmap['labelHem']).values())





    # Only apply gap to halfway and last ideogram
    gapApply = np.zeros(L)
    gapApply[9] = 1
    gapApply[19] = 1

    #pdb.set_trace()

    ideogram_length=2*np.pi*np.asarray(row_sum)/sum(row_sum)-gap*gapApply

    ideo_ends=get_ideogram_ends_diffhem(ideogram_length, gap,start = np.pi/2)

    #z=make_ideogram_arc(1.3, [11*np.pi/6, np.pi/17],rot=0.5)

    labels=list(Counter(netmap['labelHem']).keys()) 


    #pdb.set_trace()

    ideograms=[]
    shape_append=[]
    for k in range(len(ideo_ends)):
        z= make_ideogram_arc(1.0, ideo_ends[k])
        zi=make_ideogram_arc(1.1, ideo_ends[k])
        m=len(z)
        n=len(zi)


        if k < 10:
            ideograms.append(plotly.graph_objs.Scatter(x=z.real,
                                     y=z.imag,
                                     mode='lines',
                                     line=dict(color=ideo_colors[k], shape='spline', width=10),
                                     text=labels[k]+'<br>'+'{:d}'.format(row_sum[k]),
                                     hoverinfo='text',
                                     name=labels[k].split('_')[0],
                                     showlegend=True
                                     )
                             )
        else:
            ideograms.append(plotly.graph_objs.Scatter(x=z.real,
                                     y=z.imag,
                                     mode='lines',
                                     line=dict(color=ideo_colors[k], shape='spline', width=10),
                                     text=labels[k]+'<br>'+'{:d}'.format(row_sum[k]),
                                     hoverinfo='text',
                                     name=labels[k].split('_')[0],
                                     showlegend=False
                                     )
                             )

        path='M '
        for s in range(m):
            path+=str(z.real[s])+', '+str(z.imag[s])+' L '

        Zi=np.array(zi.tolist()[::-1])

        for s in range(m):
            path+=str(Zi.real[s])+', '+str(Zi.imag[s])+' L '
        path+=str(z.real[0])+' ,'+str(z.imag[0])

        
        #layout['shapes'].append(make_ideo_shape(path,'rgb(150,150,150)' , ideo_colors[k]))
        shape_append.append(make_ideo_shape(path,'rgb(150,150,150)' , ideo_colors[k]))



    ##############################################################



    print('Make axis and layout')
    axis = dict(showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title=''
                )

    layout = plotly.graph_objs.Layout(showlegend=legend,
                                      autosize=False,
                                      width=pltX,
                                      height=pltY,
                                      paper_bgcolor='rgba(255,255,255, 1)',
                                      plot_bgcolor='rgba(255,255,255, 1)',
                                      xaxis=plotly.graph_objs.XAxis(axis),
                                      yaxis=plotly.graph_objs.YAxis(axis),
                                      margin=plotly.graph_objs.Margin(l=40,
                                                                      r=40,
                                                                      b=85,
                                                                      t=100,
                                                                      ),
                                      hovermode='closest',
                                      legend=dict(x=-.2, y=1.3),
                                      font=dict(size=15)
                                        )

    layout['shapes'] = shape_append


    colorMarker = plotly.graph_objs.scatter.Marker(colorscale=[[c / 100.0, 'rgba({}, {}, {}, {})'.format(*color_range(c / 100.0))] for c in range(101)], size=1, color=my_weights, showscale=True, colorbar=dict(tickfont=dict(size=20)))

    color_trace = plotly.graph_objs.Scatter(x=[0 for _ in my_weights],y=[0 for _ in my_weights],mode='markers',marker=colorMarker,showlegend=False)

    #data = plotly.graph_objs.Data([color_trace] + lines + edge_info + [trace2])
    print('Make data object')
    data = plotly.graph_objs.Data(lines+edge_info+ideograms+[color_trace])
    #data = plotly.graph_objs.Data(ideograms)
    print('Make fig object')
    fig = plotly.graph_objs.Figure(data=data, layout=layout)
    fig.update_layout(legend_font_size=22, legend = dict(bgcolor = 'rgba(0,0,0,0)', orientation="h"))

    print('Write image')
    fig.write_image(opname)


