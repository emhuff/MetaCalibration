#!/user/bin/env python
import sys
import numpy as np
import matplotlib.pyplot as plt


def dict_list_to_recarray(items):
    """
    @brief Take a loose list of dictionaries, and turn them into a recarray with
    fields defined as 'object' for clarity
    @param items Iterable collection of dictionaries
    @author Eric Cousineau <eacousineau@gmail.com>
    @note xref: http://docs.scipy.org/doc/numpy/reference/generated/numpy.core.records.fromrecords.html
    """
    # Collect the unique fields
    fields_raw = []
    for item in items:
        fields_raw += item.keys()
    fields = set(fields_raw)
    # Get value sets, replacing empty fields with 'None'
    rec_list = []
    for item in items:
        values = [item.get(field, None) for field in fields]
        rec_list.append(values)
    # Construct the np.recarray
    return np.core.records.fromrecords(rec_list, names=','.join(fields))

def get_results():
    results = [{'branch':'CGC','algorithm':'regauss','MC':True,'m1':3.6,'m1err':7.2,'m2':2.3,'m2err':6.4,'a1':-20.6,'a1err':4.6,'a2':-19.4,'a2err':3.6,'c1':0.0,'c1err':.2,'c2':-0.1,'c2err':0.2},
           {'branch':'CGC','algorithm':'regauss','MC':False,'m1':71.9,'m1err':6.1,'m2':65.5,'m2err':4.9,'a1':-39.7,'a1err':3.8,'a2':-43.9,'a2err':2.9,'c1':0.0,'c1err':0.1,'c2':0.0,'c2err':0.1},
           {'branch':'CGC','algorithm':'regauss-noisy','MC':True,'m1':-15.0,'m1err':8.9,'m2':3.7,'m2err':7.9,'a1':-24.5,'a1err':5.7,'a2':-19.7,'a2err':4.5,'c1':-0.1,'c1err':0.2,'c2':-0.1,'c2err':0.2},
           {'branch':'CGC','algorithm':'regauss-noisy','MC':False,'m1':117.2,'m1err':7.7,'m2':105.2,'m2err':6.6,'a1':40.2,'a1err':4.9,'a2':-48.4,'a2err':4.1,'c1':0.2,'c1err':0.2,'c2':-0.1,'c2err':0.2},
           {'branch':'CGC','algorithm':'ksb','MC':True,'m1':8.0,'m1err':9.8,'m2':17.0,'m2err':8.7,'a1':-19.8,'a1err':6.0,'a2':-9.2,'a2err':5.0,'c1':0.3,'c1err':0.2,'c2':-0.2,'c2err':0.2},
           {'branch':'CGC','algorithm':'ksb','MC':False,'m1':132.2,'m1err':8.8,'m2':145.5,'m2err':10.4,'a1':114.8,'a1err':5.4,'a2':109.6,'a2err':6.1,'c1':-0.4,'c1err':0.2,'c2':0.0,'c2err':.3},
           {'branch':'CGC','algorithm':'moments','MC':True,'m1':41.0,'m1err':17.5,'m2':50.3,'m2err':18.3,'a1':-85.3,'a1err':9.9,'a2':-82.6,'a2err':9.5,'c1':0.1,'c1err':0.4,'c2':-0.2,'c2err':0.5},
           {'branch':'CGC','algorithm':'moments','MC':False,'m1':3223.8,'m1err':138.2,'m2':3223.8,'m2err':173.1,'a1':4480.5,'a1err':77.6,'a2':4604.4,'a2err':88.8,'c1':0.9,'c1err':3.4,'c2':-0.8,'c2err':4.6},
           {'branch':'RGC','algorithm':'regauss','MC':True,'m1':-5.3,'m1err':7.8,'m2':6.1,'m2err':6.6,'a1':-3.6,'a1err':4.0,'a2':-3.1,'a2err':3.7,'c1':0.1,'c1err':0.2,'c2':0.1,'c2err':0.2},
           {'branch':'RGC','algorithm':'regauss','MC':False,'m1':30.4,'m1err':5.2,'m2':24.9,'m2err':5.0,'a1':-29.5,'a1err':2.8,'a2':-18.6,'a2err':2.8,'c1':0.0,'c1err':0.1,'c2':0.2,'c2err':0.1},
           {'branch':'CGC-Noaber','algorithm':'regauss','MC':True,'m1':7.4,'m1err':6.9,'m2':6.1,'m2err':6.4,'a1':-33.9,'a1err':12.7,'a2':-23.7,'a2err':11.2,'c1':0.1,'c1err':0.2,'c2':0.1,'c2err':0.2},
           {'branch':'CGC-Noaber','algorithm':'regauss','MC':False,'m1':40.7,'m1err':2.9,'m2':-43.9,'m2err':2.9,'a1':-27.4,'a1err':5.2,'a2':-26.5,'a2err':5.0,'c1':0.1,'c1err':0.1,'c2':0.1,'c2err':0.1},
           {'branch':'RGC-Noaber','algorithm':'regauss','MC':True,'m1':1.3,'m1err':5.9,'m2':4.5,'m2err':6.4,'a1':-11.5,'a1err':11.4,'a2':-.8,'a2err':12.2,'c1':0.0,'c1err':0.2,'c2':-.1,'c2err':0.2},
           {'branch':'RGC-Noaber','algorithm':'regauss','MC':False,'m1':16.4,'m1err':3.0,'m2':17.4,'m2err':3.4,'a1':2.2,'a1err':5.8,'a2':2.5,'a2err':6.4,'c1':0.2,'c1err':0.1,'c2':0.0,'c2err':0.1},
           {'branch':'RGC-FixedAber','algorithm':'regauss','MC':True,'m1':-11.6,'m1err':8.9,'m2':-14.2,'m2err':7.5,'a1':-17.4,'a1err':22.6,'a2':-18.3,'a2err':18.6,'c1':0.2,'c1err':0.2,'c2':-0.2,'c2err':0.2},
           {'branch':'RGC-FixedAber','algorithm':'regauss','MC':False,'m1':61.6,'m1err':6.6,'m2':63.7,'m2err':5.2,'a1':-30.0,'a1err':12.1,'a2':-32.3,'a2err':13.5,'c1':0.3,'c1err':0.2,'c2':0.0,'c2err':0.1}]
    #results_array = dict_list_to_recarray(results)
    return np.array(results)



def main(argv):
    results = get_results()
    
    unique_id = np.unique([thing['branch']+'-'+thing['algorithm'] for thing in results])
    is_mc = [thing['MC'] for thing in results]
    bralgorithm = [thing['branch']+'-'+thing['algorithm'] for thing in results]


    yvalues = 2*(np.arange(len(unique_id)+1))
    yticklabels = list(unique_id)
    yticklabels.insert(0,' ')
    
    fig1,ax1 = plt.subplots(figsize=(10,10))
    #fig2,ax2 = plt.subplots()
    #fig3,ax3 = plt.subplots()
    #fig4,ax4 = plt.subplots()
    ax1.set_xscale('linear')
    #ax1.set_ylim(-1000,1000) # useful range for m1,m2
    ax1.set_xlim(-100,200) # useful range for m1,m2    
    ax1.set_xlabel(r'multiplicative biases ($m_1$, $m_2$) $\times 10^{3}$')
    ax1.set_yticks(yvalues)
    ax1.set_yticklabels(yticklabels)
    ax1.set_ylim(0,np.max(yvalues)+1)
    for i,this_id in enumerate(unique_id):
        index = 2*(1+i)
        this_mc  = (np.array(bralgorithm) == this_id) &  np.array(is_mc)
        this_raw = (np.array(bralgorithm) == this_id) & ~np.array(is_mc)
        # plot the effects on m.

        ax1.errorbar([results[this_raw][0]['m2'],results[this_mc][0]['m2']],[index-.25,index-0.1],
                     xerr= [2*results[this_raw][0]['m2err'],2*results[this_mc][0]['m2err']],
                     linestyle=' ',mfc=(1, 1, 0, 0.5),ecolor='b',capsize=1.5,alpha=0.75)
        ax1.errorbar([results[this_raw][0]['m1'],results[this_mc][0]['m1']],[index+.25,index+0.1],
                     xerr = [2*results[this_raw][0]['m1err'],2*results[this_mc][0]['m1err']],
                     linestyle=' ',mfc=(1, 1, 0, 0.5),ecolor='b',capsize=1.5,alpha=0.75)
        ax1.errorbar([results[this_raw][0]['m2'],results[this_mc][0]['m2']],[index-.25,index-0.1],
                     xerr = [results[this_raw][0]['m2err'],results[this_mc][0]['m2err']],
                     linestyle=' ',marker='.',mfc='y',mec='y',ecolor='b',alpha=0.75)
        ax1.errorbar([results[this_raw][0]['m1'],results[this_mc][0]['m1']],[index+.25,index+0.1],
                     xerr = [results[this_raw][0]['m1err'], results[this_mc][0]['m1err']],
                     linestyle=' ',marker='.',mfc='y',mec='y',ecolor='b',alpha=0.75)

        ax1.plot([results[this_raw][0]['m1']],[index+0.25],'s',mfc='w',mec='k',markeredgewidth=0.75)
        ax1.plot([results[this_raw][0]['m2']],[index-0.25],'o',mfc='w',mec='k',markeredgewidth=0.75)
        ax1.plot([results[this_mc][0]['m1']],[index+0.1],'s',mfc='k',mec='k',markeredgewidth=0.75)
        ax1.plot([results[this_mc][0]['m2']],[index-0.1],'o',mfc='k',mec='k',markeredgewidth=0.75)
        
        if results[this_raw][0]['m1'] > 200:
            upper1 = 200
        else:
            upper1 = results[this_raw][0]['m1']
            
        if results[this_raw][0]['m2'] > 200:
            upper2 = 200
        else:
            upper2 = results[this_raw][0]['m2']
        
        m1sep = np.abs(ax1.transData.transform((upper1,1))[0] - ax1.transData.transform((results[this_mc][0]['m1'],1))[0])
        m2sep = np.abs(ax1.transData.transform((upper2,1))[0] - ax1.transData.transform((results[this_mc][0]['m2'],1))[0])

        barfrac1 = 15/m1sep
        barfrac2 = 15/m2sep
        print 'seps:',m1sep,m2sep
        if ((results[this_raw][0]['m1']) > (results[this_mc][0]['m1'])):
            connectionstyle = 'bar,fraction='+str(barfrac1)
        else:
            connectionstyle = 'bar,fraction=-'+str(barfrac1)
            
        ax1.annotate("",
                     xytext= (upper1, index+.27),
                     xy= ( results[this_mc][0]['m1'], index+.27),
                     arrowprops=dict(arrowstyle="->",alpha=0.3,connectionstyle=connectionstyle) )
        if ((results[this_raw][0]['m2']) > (results[this_mc][0]['m2'])):
            connectionstyle = 'bar,fraction=-'+str(barfrac2)
        else:
            connectionstyle = 'bar,fraction='+str(barfrac2)
        ax1.annotate("",
                     xytext= ( upper2, index-.27),
                     xy= ( results[this_mc][0]['m2'],index-.27),
                     arrowprops=dict(arrowstyle="->",alpha=0.3,connectionstyle=connectionstyle) )
        

        #ax1.axvline(0,color='red',linestyle='--',alpha=0.15)
        ax1.axvspan(-2,2,alpha=0.01,facecolor='red')
    fig1.tight_layout()
    fig1.savefig("m_results_linear.pdf",format='pdf')           
    pass



 
if __name__ == "__main__":
    import pdb, traceback
    try:
        main(sys.argv)
    except:
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

