function model_compute,model,x
  inner = where(x lt max(model.x) AND x gt min(model.x),ct_inner)
  outer_hi = where(x ge max(model.x),ct_outer_hi)
  outer_lo = where(x le min(model.x),ct_outer_lo)
  y_out = x*0d
  if ct_inner gt 0 then $
     y_out[inner] = interpol(model.y,model.x,x[inner])
  if ct_outer_hi gt 0 then $
     y_out[outer_hi] = model.norm_upper * (x[outer_hi])^(-model.p_upper)
  if ct_outer_lo gt 0 then $
     y_out[outer_lo] = model.norm_lower * abs(x[outer_lo])^(-model.p_lower)
  return,y_out
end


function model_initialize,pts_in,bad=bad,p_tol=p_tol
; If we have objects which should not be included in the model, then
; set the 'bad' flag; all data points exactly equal to bad will be
; ignored in the analysis.
if n_elements(bad) gt 0 then begin
   pts = pts_in[where(pts_in ne bad)]
endif else pts = pts_in

npts = n_elements(pts)
sigma = max([robust_sigma(pts)/sqrt(npts),0.001])


; Work out where the power-law wings start.
if n_elements(p_tol) eq 0 then p_tol = 0.05
diff = 1.
ub = 1.
p_upper = -1
while diff gt p_tol do begin
   ub = ub + 0.1
   p_upper_new = powerlaw_index_mle(pts[where(pts gt ub)],ub)
   diff = abs(p_upper_new - p_upper)
   p_upper = p_upper_new
endwhile
; Upper bound on core is now ub

lb = -1.
p_lower = -1.
diff = 1.
while diff gt p_tol do begin
   lb = lb - 0.1
   p_lower_new = powerlaw_index_mle(abs(pts[where(pts lt lb)]),abs(lb))
   diff = abs(p_lower_new - p_lower)
   p_lower = p_lower_new
endwhile

ngrid = 1000.
x = lb + (ub-lb)*findgen(ngrid)/float(ngrid-1)
y = dblarr(ngrid)

for i = 0L,ngrid-1 do begin
   y[i] = (total(exp(-(pts-x[i])^2/2./sigma^2)/sqrt(2*!Pi*sigma^2),/double) + $
          total(exp(-(-pts-x[i])^2/2./sigma^2)/sqrt(2*!Pi*sigma^2),/double))/2.
endfor


ynorm =  (total((pts gt lb) AND (pts lt ub),/double)/float(npts))
y = y/int_tabulated(x,y) * ynorm

;Finally, get the amplitude of the power-law component.
; Normalize each power law, then multiply by the fraction of pts in
; that range.
norm_upper = ub^(p_upper-1) * total(pts gt ub,/double)/float(npts)
norm_lower = abs(lb)^(p_lower-1) * total(pts lt lb,/double)/float(npts)
model = create_struct('x',x,'y',y,'p_upper',p_upper,'p_lower',p_lower,'norm_upper',norm_upper,'norm_lower',norm_lower)

return,model
end


function em_shear_estimate,g_initial,model_e1,model_e2,cat,converge=converge,$
                           r1=R1_avg, r2=R2_avg, c1=c1, c2=c2, a1=R1psf, a2=R2psf

g = g_initial
L_thresh = 0.
print,'Running EM algorithm, monitor for convergence:'

diff = 1.
n_iter= 0 
g1_series = g_initial[0]
n_retry = 0 
R1psf = mean(cat.a1)
R2psf = mean(cat.a2)
psf_e1 = mean(cat.psf_e1)
psf_e2 = mean(cat.psf_e2)
c1 = median(cat.c1)
c2 = median(cat.c2)

while (diff gt 1e-3) AND (n_iter lt 500) do begin
   n_iter++
   ;Use g_inital to calculate new 'unlensed' ellipticities.
   e1 = cat.e1 - (g[0]*cat.R1  + R1psf * psf_e1)
   e2 = cat.e2 - (g[1]*cat.R2  + R2psf * psf_e2)


   ; But! Assign zero weight to points of extremely low likelihood.

   L1 = (model_compute(model_e1,e1))
   L2 = (model_compute(model_e2,e2))
   cut1 = where(L1 lt L_thresh,ctcut1)
   cut2 = where(L2 lt L_thresh,ctcut2)
   if ctcut1 gt 0 then L1[cut1] = 0.
   if ctcut2 gt 0 then L2[cut2] = 0.

   ; Then do a likelihood-weighted average of the responsivities.
   

   R1_avg = total(L1 * cat.R1,/double) / total(L1,/double)
   R2_avg = total(L2 * cat.R2,/double) / total(L2,/double)

   R1psf = total(L1 * cat.a1, /double) / total(L1,/double)
   R2psf = total(L2 * cat.a2, /double) / total(L2,/double)

   e1_psf_correction = total(L1 * cat.a1 * cat.psf_e1)/total(L1)
   e2_psf_correction = total(L2 * cat.a2 * cat.psf_e2)/total(L2)

   c1 = total(L1 * cat.c1,/double) / total(L1,/double)
   c2 = total(L2 * cat.c2,/double) / total(L2,/double)

   ; And use these to compute a new guess for the shear.
   g1_new = total(L1 * (cat.e1 - c1 - e1_psf_correction) )/ total(L1) / R1_avg
   g2_new = total(L2 * (cat.e2 - c2 - e2_psf_correction) )/ total(L2) / R2_avg
   ;g1_new = total(L1 * (cat.e1 - cat.c1 - cat.a1*cat.psf_e1)/cat.R1 )/ total(L1)
   ;g2_new = total(L2 * (cat.e2 - cat.c2 - cat.a2*cat.psf_e2)/cat.R2 )/ total(L2)
   ;g1_new = total(L1 * (cat.e1 - cat.c1 - cat.a1*cat.psf_e1)/cat.R1 )/ total(L1) / median(cat.r1)
   ;g2_new = total(L2 * (cat.e2 - cat.c2 - cat.a2*cat.psf_e2)/cat.R2 )/ total(L2) / median(cat.r2)
   

   diff = max([abs(g1_new-g[0]),abs(g2_new-g[1])])
;   if (n_iter eq 249 AND n_retry lt 3) then begin
   if (n_iter eq 499 AND n_retry lt 4) then begin
      g = [mean([g1_new,g[0]]),mean([g2_new,g[1]])]
;     This is a defense against limit cycles, which seem to be very common here.
      n_iter = 0
      n_retry = n_retry+1
   endif else $
      g = [g1_new,g2_new]
   if n_iter mod 10 eq 0 then print,g[0],g[1]              ;Watch for convergence.
   g1_series = [g1_series,g1_new]

endwhile
converge = 1
if ((n_retry gt 4) OR  (sqrt(g[0]^2 + g[1]^2) gt 1.)) OR (( ~finite(g[0]) ) OR (~finite(g[1]) ) )  then converge=0
if ( (sqrt(g[0]^2 + g[1]^2) gt 1.) OR (( ~finite(g[0]) ) OR (~finite(g[1]) ) ) ) then converge=0
return,g
end

function em_shear_estimate_simple, cat, r1=r1, r2 = r2, c1 = c1, c2 = c2, a1 = r1psf, a2 = r2psf, psf1=psf1,psf2=psf2, weight1=w1,$
  weight2=w2
if n_elements(w1) eq 0 then w1 = replicate(1.,n_elements(cat))
if n_elements(w2) eq 0 then w2 = replicate(1.,n_elements(cat))

R1 = total(cat.R1*w1)/total(w1)
R2 = total(cat.R2*w2)/total(w2)
R1psf = total(cat.a1*w1)/total(w1)
R2psf = total(cat.a2*w2)/total(w2)
c1 = total(cat.c1*w1)/total(w1)
c2 = total(cat.c2*w2)/total(w2)
psf1 = total(cat.psf_e1*w1)/total(w1)
psf2 = total(cat.psf_e2*w2)/total(w2)

;g1 = total(( cat.e1 - R1psf*cat.psf_e1/2. - cat.c1 )/R1 * w1)/total(w1) 
;g2 = total(( cat.e2 - R2psf*cat.psf_e2/2. - cat.c2 )/R2 * w2)/total(w2) 
g1 = total( ( cat.e1 - R1psf*cat.psf_e1/2. - cat.c1 )/R1  * w1) / total(w1)
g2 =  total( ( cat.e2 - R2psf*cat.psf_e2/2. - cat.c2 )/R2 * w2) / total(w2)

g = [g1,g2]
return,g
end


function select_for_analysis,cat
  keep = where(cat.e1 ne -10 AND cat.c1 ne -10 AND $
               cat.e2 ne -10 AND cat.c2 ne -10 AND $
               cat.r1 ne -10 AND cat.r2 ne -10 AND $
               cat.weight gt 0 AND $
               abs(cat.e1) lt 100 AND abs(cat.e2) lt 100,ct)
  return,keep
end


pro em_shear_calib
;common ellipticity_prior_models,model_e1,model_e2,cat
; Go and get the shear calibration files.
;template = "../Great3/Outputs-Moments/cgc_metacal_moments-*.fits"
;template = "../Great3/Outputs-Regauss/cgc_metacal_regauss_fix-*.fits"
;template = "../Great3/Outputs-Regauss-SymNoise/cgc_metacal_symm-*.fits"
;template = "../Great3/Outputs-Regauss-NoAber/cgc_noaber_metacal-*.fits"
template = "../Great3/Outputs-Regauss-NoAber-SymNoise/cgc_noaber_metacal_symm-*.fits"
;template = "../Great3/Outputs-KSB/output_catalog-*.fits"
;template = "../Great3/Outputs/Control-Ground-Constant/output_catalog-*.fits"
;template = "../Great3/Outputs-Real-Regauss/output_catalog*.fits"
catfiles = file_search(template,count=ct)
print,'Found ',ct,' files. Loading data.'
cat = mrdfits(catfiles[0],1,/silent)

for i = 1,ct-1 do cat = [cat,mrdfits(catfiles[i],1,/silent)]

cat = jjadd_tag(cat,'e1',cat.g1)
cat = jjadd_tag(cat,'e2',cat.g2)

print,' Removing bad shape measurements.'
keep = select_for_analysis(cat)
cat = cat[keep]

; Construct the global ellipticity prior. We could try to deconvolve a
; shear prior from this if we wanted to, but we don't really want to.

print, 'Constructing global shape prior.'
print, 'Removing our estimate of the constant shape measurement biases first.'

; To build the prior, assume that the correct shear is zero.
; Then subtract off what we think the effects of psf and additive bias
; are.
e1_prior = (cat.e1 - cat.a1 * cat.psf_e1 - cat.c1)
e2_prior = (cat.e2 - cat.a2 * cat.psf_e2 - cat.c2)


model_e1 = model_initialize(e1_prior ,bad=-10)
model_e2 = model_initialize(e2_prior ,bad=-10)


;Check to see if the priors make sense.
z = -10 + 20*findgen(1000)/999.
y = model_compute(model_e1,z)
psopen,'prior-regauss-noaber-symn-shear',xsize=6,ysize=6,/inches
prepare_plots,/color
plot,z,y,/ylog,xr=[-20,20],thick=3
peak = max(model_e1.y)
plothist,cat.e1,bin=0.01,xr=[-10,10],/ylog,peak=peak,/overplot,color=200,yr=[0.1*model_compute(model_e1,10),1.],title='e1 prior'
oplot,z,y,thick=4
y = model_compute(model_e2,z)
prepare_plots,/color
plot,z,y,/ylog,xr=[-20,20],thick=3
peak = max(model_e2.y)
plothist,cat.e2,bin=0.01,xr=[-10,10],/ylog,peak=peak,/overplot,color=200,yr=[0.1*model_compute(model_e2,10),1.],title='e2 prior'
oplot,z,y,thick=4



prepare_plots,/reset
psclose
; Use Expectations-Maximization to estimate the shear on a per-field
; basis.
g_initial = [0.,0.]

field_shear = dblarr(ct,2)
mean_shear = dblarr(ct,2)
id = lonarr(ct)
converged = fltarr(ct)
r1 = fltarr(ct)
r2 = fltarr(ct)
a1psf = fltarr(ct)
a2psf = fltarr(ct)
c1 = fltarr(ct)
c2 = fltarr(ct)
psf_e1 = fltarr(ct)
psf_e2 = fltarr(ct)

; We're going to assess how well each field corresponds to the prior.
; Take a sample of the (symmetrized) shape distribution.
nsample = 500000
ks_e1 = [cat.e1,-cat.e1]
ks_e2 = [cat.e2,-cat.e2]
ks_e1 = ks_e1[random_indices(n_elements(ks_e1),nsample)]
ks_e2 = ks_e2[random_indices(n_elements(ks_e2),nsample)]


ksstat1 = fltarr(ct)
ksstat2 = fltarr(ct)
outlierFrac1 = fltarr(ct)
outlierFrac2 = fltarr(ct)
outlierThresh = 3.
; Make plots showing how the distribution shifts when shear is
; applied.

;readcol,'cgc-truthtable.txt',id_true,g1,g2
readcol,'cgc-noaber-truthtable.txt',id_true,g1,g2

psopen,'metacal-regauss-noaber-symn-distributions',xsize=8,ysize=5,/inches,/color
prepare_plots,/color
badfields = [93,97,150,160]


for i = 0,ct-1 do begin
   this_catalog = mrdfits(catfiles[i],1,/silent)
   id[i] = long((stregex(catfiles[i],'-([0-9]{3}).fits',/sub,/extract))[1])
   this_catalog = jjadd_tag(this_catalog,'e1',this_catalog.g1)
   this_catalog = jjadd_tag(this_catalog,'e2',this_catalog.g2)
   this_keep = select_for_analysis(this_catalog)
   this_catalog = this_catalog[this_keep]
   print,'Using EM and the prior to estimate the shear for field: '+(stregex(catfiles[i],'-([0-9]*).fits',/sub,/ext))[1]

   mean_shear[i,*] = [mean(this_catalog.e1),mean(this_catalog.e2)]
   g = em_shear_estimate(g_initial,model_e1,model_e2,this_catalog,converge=this_converged, $
                         r1=this_r1, r2=this_r2, c1=this_c1, c2=this_c2, a1=this_a1, a2=this_a2)
;   L1 = model_compute(model_e1, this_catalog.e1)
;   L2 = model_compute(model_e2, this_catalog.e2)
;   g = em_shear_estimate_simple(this_catalog, r1=this_r1, r2=this_r2, c1=this_c1, c2=this_c2, $
;                                a1=this_a1, a2=this_a2, psf1=psf1,psf2=psf2, weight1=this_catalog.weight,$
;                                weight2 = this_catalog.weight)
   psf_e1[i] = mean(this_catalog.psf_e1)
   psf_e2[i] = mean(this_catalog.psf_e2)
   r1[i] = this_r1
   r2[i] = this_r2
   a1psf[i] = this_a1
   a2psf[i] = this_a2
   c1[i] = this_c1
   c2[i] = this_c2
   converged[i] = this_converged
   field_shear[i,0] = g[0]
   field_shear[i,1] = g[1]
   
   

;--------------------------------------------------
;  Make plots.   
; First, plot the prior.
   y = (model_compute(model_e1,z)  + model_compute(model_e1,-z))/2.
;  Then plot the uncorrected shape distribution
   peak = max(y)
   xb = 3 ; min/max on x-axis for histogram
   plothist,this_catalog.e1,bin=0.05,xr=[-xb,xb],/ylog,peak=peak,title='e1, field '+(stregex(catfiles[i],'-([0-9]*).fits',/sub,/ext))[1],yr=[1e-4,1]
   oplot,z,y,color=50
;  Finally, plot the unsheared shape distribution
   e1_unsheared = (this_catalog.e1 - this_catalog.r1 * g[0] - this_a1*psf_e1[i] - this_c1)
   e2_unsheared = (this_catalog.e2 - this_catalog.r2 * g[1] - this_a2*psf_e2[i] - this_c2)
   plothist,e1_unsheared,bin=0.05,xr=[-xb,xb],/ylog,peak=peak,/overplot,color=120,yr=[1e-4,1]
   e1_best = (this_catalog.e1 - this_catalog.r1 * g1[i] - this_a1*psf_e1[i] - this_c1)
   plothist,e1_best,bin=0.05,xr=[-xb,xb],/ylog,peak=peak,/overplot,color=200,yr=[1e-4,1]
   legend,['prior','raw','estimated','true'],color=[50,255,120,200],box=0,/top,/right,charsize=1.,line=0
;   legend,['prior','raw','estimated'],color=[50,255,120],box=0,/top,/right,charsize=1.,line=0
   legend,['g_true = '+string(g1[i],form='(F0.4)'), 'g_est = '+string(g[0],form='(F0.4)'),'!7D!6 = '+string(g[0]-g1[i],form='(F0.4)')],/top,/left,box=0,charsize=.75
;--------------------------------------------------   
   e1test = [(this_catalog.e1 - this_catalog.r1 * g[0] - this_a1*psf_e1[i] - this_c1)]
   e2test = [(this_catalog.e2 - this_catalog.r2 * g[1] - this_a2*psf_e2[i] - this_c2)]
   outlierFrac1[i] = total(abs(e1_unsheared) gt outlierThresh) / float(n_elements(e1_unsheared) )
   outlierFrac2[i] = total(abs(e2_unsheared) gt outlierThresh) / float(n_elements(e2_unsheared) )

   kstwo,e1test,ks_e1,ks1,ksp1
   kstwo,e2test,ks_e2,ks2,ksp2
   ksstat1[i] = ksp1
   ksstat2[i] = ksp2

   ;ksstat1[i] = variance(e1test)/variance(ks_e1)
   ;ksstat2[i] = variance(e2test)/variance(ks_e2)
   print,'Estimated shear (g1, g2) is:', g[0], g[1]

endfor
psclose

forprint,text='Great3-metaCal-CGC-regauss-noaber-symn.txt',id,field_shear[*,0],field_shear[*,1],converged,/nocomment



;readcol,'cgc-truthtable.txt',id_true,g1,g2
readcol,'cgc-noaber-truthtable.txt',id_true,g1,g2

forprint, text = 'metaCal-outlier-diagnostics-regauss-noaber-symn.txt', id, field_shear[*,0], g1, psf_e1, field_shear[*,1], g2, psf_e2, $
          converged, ksstat1, ksstat2, outlierFrac1, outlierFrac2, width = 1000, comment = "id   g1 (est)    g1 (true)     psf e1    g2 (est)    g2 (true)    psf e2    converged    ks1    ks2    outlierFrac (|g1|>2)  outlierFrac (|g2|>2)"


;use = where( ( converged eq 1) AND (ksstat1 le 1.01) AND (ksstat2 lt 1.01) )
;use = where( ( converged eq 1) AND (ksstat1 gt 1.e-5) AND (ksstat2 gt 1e-5) )
use = where( ( converged eq 1) )
id_cut = id[where( (( converged eq 1) AND (ksstat1 gt 1.e-5) AND (ksstat2 gt 1e-5)) eq 0) ]
id = id[use]
field_shear_cut = field_shear(where(use eq 0 ),* )

field_shear = field_shear[use,*]



match,id_true,id,ind_true,ind_mc
match,id_true,id_cut, ind_cut, ind_mn
psopen,'metaCalResults-regauss-noaber-symn',xsize=8,ysize=8,/inches,/color
prepare_plots,/color

coeff1 = linfit(g1[ind_true],field_shear[ind_mc,0],y=y1)
coeff2 = linfit(g2[ind_true],field_shear[ind_mc,1],y=y2)

; What's the scatter around this?
if total(~finite(y1)) eq 0 then begin
   histogauss,field_shear[ind_mc,0]-y1,a1;,/noplot
   coeff1 = linfit(g1[ind_true],field_shear[ind_mc,0],y=y1,measure_err=replicate(a1[2],n_elements(ind_true)),sigma=sigma)
   plot,g1[ind_true],field_shear[ind_mc,0],ps=1,xtitle='g_1 (true)', ytitle='g_1 (recovered)'
   oplot,g1[ind_cut], field_shear[ind_mn,0],ps=7,color=50
   oplot,[-1,1],[-1,1],color=200,line=2
   xyouts,0.3,0.2,string(form='("b, m = ",F0," ",F0,"  !9 + !6  ",F0," ",F0 )',coeff1[0],coeff1[1],sigma[0],sigma[1]),/norm,charsize=1.
   plot,g1[ind_true],field_shear[ind_mc,0] - g1[ind_true],ps=1,xtitle='g_1 (true)', ytitle='g_1 (recovered) - g_1 (true)'
   oplot,g1[ind_cut], field_shear[ind_mn,0] - g1[ind_cut],ps=7,color=50
   hline,0,color=200,line=2
   oplot,g1[ind_true],y1-g1[ind_true],color=75,line=3
   xyouts,0.3,0.2,string(form='("b, m = ",F0," ",F0,"  !9 + !6  ",F0," ",F0 )',coeff1[0],coeff1[1],sigma[0],sigma[1]),/norm,charsize=1.
endif

   coeff2 = linfit(g2[ind_true],field_shear[ind_mc,1],y=y2)
if total(~finite(y2)) eq 0. then begin
   histogauss,field_shear[ind_mc,1]-y2,a2;,/noplot
   coeff2 = linfit(g2[ind_true],field_shear[ind_mc,1],y=y2,measure_err=replicate(a2[2],n_elements(ind_true)),sigma=sigma)

   plot,g2[ind_true],field_shear[ind_mc,1],ps=1,xtitle='g_2 (true)', ytitle='g_2 (recovered)'
   oplot,g2[ind_cut], field_shear[ind_mn,1],ps=7,color=50
   oplot,[-1,1],[-1,1],color=200,line=2
   xyouts,0.3,0.2,string(form='("b, m = ",F0," ",F0,"  !9 + !6  ",F0," ",F0 )',coeff2[0],coeff2[1],sigma[0],sigma[1]),/norm,charsize=1.
   plot,g2[ind_true],field_shear[ind_mc,1] - g2[ind_true],ps=1,xtitle='g_2 (true)', ytitle='g_2 (recovered) - g_2 (input)'
   hline,0,color=200,line=2
   oplot,g2[ind_true],y2-g2[ind_true],color=75,line=3
   oplot,g2[ind_cut], field_shear[ind_mn,1] - g2[ind_cut],ps=7,color=50
   xyouts,0.3,0.2,string(form='("b, m = ",F0," ",F0,"  !9 + !6  ",F0," ",F0 )',coeff2[0],coeff2[1],sigma[0],sigma[1]),/norm,charsize=1.

endif
psclose
prepare_plots,/reset

; Use Hogg's iterative linear least-squarefit to do a
; sigma-clipped linear fit.

aa1  = [[g1[ind_true]],[replicate(1.,n_elements(ind_true))]]
yy1 = field_shear[ind_mc,0]
ww1 = replicate(1.,n_elements(ind_true))

aa2 = [[g2[ind_true]],[replicate(1.,n_elements(ind_true))]]
yy2 = field_shear[ind_mc,1]
ww2 = replicate(1.,n_elements(ind_true))

hogg_iter_linfit,aa1,yy1,ww1,xx1,covar=covar1, nsigma=5.
hogg_iter_linfit,aa2,yy2,ww2,xx2,covar=covar2, nsigma=5.

zz1 = aa1 ## xx1
zz2 = aa2 ## xx2


psopen,'metaCalResults-regauss-noaber-symn-sigClipped',xsize=7,ysize=7,/inches,/color
prepare_plots,/color

plot,g1[ind_true],field_shear[ind_mc,0],ps=6,xtitle='g_1 (true)', ytitle='g_1 (recovered)', yr=[-0.1,0.1]
oplot,[-1,1],[-1,1],color=200,line=2
oplot,aa1[0,*], zz1, color=50,line=3
xyouts,0.2,0.2,string(form='("m, b = ",F0," ",F0,"  !9 + !6  ",F0," ",F0 )',xx1[0],xx1[1],sqrt(covar1[0,0]),sqrt(covar1[1,1])),/norm,charsize=1.
plot,g1[ind_true],field_shear[ind_mc,0] - g1[ind_true],ps=1,xtitle='g_1 (true)', ytitle='g_1 (recovered) - g_1 (true)', yr=[-0.05, 0.05]
hline,0,color=200,line=1
oplot,aa1[0,*], zz1 - g1[ind_true], color=50,line=3
xyouts,0.2,0.2,string(form='("m,b = ",F0," ",F0,"  !9 + !6  ",F0," ",F0 )',xx1[0],xx1[1],sqrt(covar1[0,0]),sqrt(covar1[1,1])),/norm,charsize=1.

plot,g2[ind_true],field_shear[ind_mc,1],ps=6,xtitle='g_2 (true)', ytitle='g_2 (recovered)', yr=[-0.1,0.1]
oplot,[-1,1],[-1,1],color=200,line=2
oplot,aa2[0,*], zz2, color=50,line=3
xyouts,0.2,0.2,string(form='("m, b = ",F0," ",F0,"  !9 + !6  ",F0," ",F0 )',xx2[0],xx2[1],sqrt(covar2[0,0]),sqrt(covar2[1,1])),/norm,charsize=1.
plot,g2[ind_true],field_shear[ind_mc,1] - g2[ind_true],ps=1,xtitle='g_2 (true)', ytitle='g_2 (recovered) - g_2 (true)', yr=[-0.05,0.05]
hline,0,color=200,line=1
oplot,aa2[0,*], zz2 - g2[ind_true], color=50,line=3
xyouts,0.2,0.2,string(form='("m, b = ",F0," ",F0,"  !9 + !6  ",F0," ",F0 )',xx2[0],xx2[1],sqrt(covar2[0,0]),sqrt(covar2[1,1])),/norm,charsize=1.

psclose

; Is there any residual psf dependence?
psopen,'metaCalResults-regauss-noaber-symn-psf_dependence',xsize=6,ysize=6,/inches,/color
prepare_plots,/color
 plot,psf_e1[ind_mc],field_shear[ind_mc,0]-g1[ind_true],ps=1,xtitle='!3psf_e1',ytitle='g_1 (measured) - g_1 (true)',xmargin=[14,4],yr=[-0.015,0.015],/ystyle

 plot,psf_e2[ind_mc],field_shear[ind_mc,1]-g2[ind_true],ps=1,xtitle='!3psf_e2',ytitle='g_2 (measured) - g_2 (true)',xmargin=[14,4],yr=[-0.015,0.015],/ystyle
psclose

; Can we predict which fields are likely to be bad by comparing them
; with the ellipticity prior?
psopen,'metaCalResults-regauss-noaber-symn-ks',xsize=8,ysize=8,/inches,/color
prepare_plots
plot,ksstat1[use[ind_mc]],field_shear[ind_mc,0]-g1[ind_true],ps=1,xtitle='(dis-)similarity to prior',ytitle='g_1 (measured) - g_1 (true)',charsize=2.,xmargin=[14,4],/xlog
vline,1e-5,color=200,line=2
plot,ksstat2[use[ind_mc]],field_shear[ind_mc,1]-g2[ind_true],ps=1,xtitle='(dis-)similarity to prior',ytitle='g_2 (measured) - g_2 (true)',charsize=2.,xmargin=[14,4],/xlog
vline,1e-5,color=200,line=2
psclose
prepare_plots,/reset

stop


end
