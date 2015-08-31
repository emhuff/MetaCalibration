function select_for_analysis,cat
  keep = where(cat.g1 ne -10 AND cat.c1 ne -10 AND $
               cat.g2 ne -10 AND cat.c2 ne -10 AND $
               cat.r1 ne -10 AND cat.r2 ne -10 AND $
               cat.weight gt 0 AND $
               abs(cat.g1) lt 100 AND abs(cat.g2) lt 100,ct)
  return,keep
end


pro metacal_diagnose

; Go get the files.
;--------------------------------------------------
  template_const = '/n/des/huff.791/Great3/Outputs-Regauss/cgc_metacal_regauss_fix*.fits'
;  template_const = '/n/des/huff.791/Great3/Outputs-Regauss-SymNoise/output_catalog-*.fits'
;  template_const = '../Great3/Outputs-Regauss-SymNoise/cgc_metacal_symm-*.fits'
  files_const = file_search(template_const,count=ctc)

  
  template_none = '/n/des/huff.791/Great3/Outputs-CGN-Regauss/output_catalog-*.fits'
  files_none = file_search(template_none,count=ctn)

; Index these against each other, since they're not necessarily
; complete.

  const_fieldno = lonarr(ctc)
  none_fieldno = lonarr(ctn)

  for i =0,ctc-1 do begin
     const_fieldno[i] = long((stregex(files_const[i],'-([0-9]{3}).fits',/sub,/ext))[1])
  endfor

  for i = 0,ctn-1 do begin
     none_fieldno[i] = long((stregex(files_none[i],'-([0-9]{3}).fits',/sub,/ext))[1])
  endfor


; Match the two file arrays against each other by field number.
  match,const_fieldno, none_fieldno,cind,nind
  files_const = files_const[cind]
  files_none = files_none[nind]
  nfiles = n_elements(nind)
  fieldnumbers = none_fieldno[nind]

; Now loop over the file list and make diagnostic plots for each one.
; Read in the shear table for the CGC branch.
  readcol,'cgc-truthtable.txt',truth_field_id,truth_g1,truth_g2;,epoch,truth_psf_g1,truth_psf_g2
  readcol,'cgc-psf-truth.txt',none, truth_psf_g1, truth_psf_g2, subfield_index
  psopen,'cgc-metacal-res_check',xsize=7,ysize=7,/inches,/color
  prepare_plots,/color
  bias1 = fltarr(nfiles)
  sigma1=  fltarr(nfiles)
  bias2 = fltarr(nfiles)
  sigma2=  fltarr(nfiles)
  g1_arr = fltarr(nfiles)
  g2_arr = fltarr(nfiles)
  psf1_arr = fltarr(nfiles)
  psf2_arr = fltarr(nfiles)


  for i = 0,nfiles-1 do begin
     ncat = mrdfits(files_none[i],1)
     ncat = ncat[select_for_analysis(ncat)]
     ccat = mrdfits(files_const[i],1)
     ccat = ccat[select_for_analysis(ccat)]
     match,ccat.id,ncat.id,cind,nind
     ccat = ccat[cind]
     ncat = ncat[nind]
; Look up the field properties.
     g1 = (truth_g1[where(truth_field_id eq fieldnumbers[i])])[0]
     g2 = (truth_g2[where(truth_field_id eq fieldnumbers[i])])[0]
     psf_g1 = (truth_psf_g1[where(truth_field_id eq fieldnumbers[i])])[0]
     psf_g2 = (truth_psf_g2[where(truth_field_id eq fieldnumbers[i])])[0]
     
     de1_dg1 = (ccat.g1 - ncat.g1)/g1
     de2_dg2 = (ccat.g2 - ncat.g2)/g2

     bias1[i] = mean(ccat.r1 - de1_dg1)
     bias2[i] = mean(ccat.r2 - de2_dg2)
     sigma1[i] = stddev(ccat.r1 - de1_dg1)
     sigma2[i] = stddev(ccat.r2 - de2_dg2)
     g1_arr[i] = g1
     g2_arr[i] = g2
     psf1_arr[i] = psf_g1
     psf2_arr[i] = psf_g2
  
     if n_elements(all_r1_true) eq 0 then begin
        all_r1_meas = ccat.r1
        all_r2_meas = ccat.r2
        all_r1_true = de1_dg1
        all_r2_true = de2_dg2
        all_weight = ccat.weight
        all_g1 = g1_arr
        all_g2 = g2_arr
     endif else begin
        all_r1_meas = [all_r1_meas, ccat.r1]
        all_r2_meas = [all_r2_meas, ccat.r2]
        all_r1_true = [all_r1_true, de1_dg1]
        all_r2_true = [all_r2_true, de2_dg2]
        all_g1 = [all_g1, g1_arr]
        all_g2 = [all_g2, g2_arr]
     endelse


;  Now make some plots.
     ;window,0
     plot,de1_dg1,ccat.r1,ps=3,xr=[-2,5],yr=[-2,5],xtitle='d(e1) / d(g1)',ytitle='MetaCal R1',title='field '+string(fieldnumbers[i],form=(I0))
     xyouts,-1.5,4.5,"g1 = "+string(g1,form=("(F0.4)"))
     oplot,[-100,100],[-100,100],thick=2,line=2,color=200
     ;window,1
     plot,de2_dg2,ccat.r2,ps=3,xr=[-2,5],yr=[-2,5],xtitle='d(e2) / d(g2)',ytitle='MetaCal R2',title='field '+string(fieldnumbers[i],form=(I0))
     oplot,[-100,100],[-100,100],thick=2,line=2,color=200
     xyouts,-1.5,4.5,"g2 = "+string(g2,form=("(F0.4)"))
     ;window,2
     ;plot,ncat.r1,ccat.r1,ps=3,xr=[-2,5],yr=[-2,5],xtitle='r1 (no shear)',ytitle='r1 (with shear)'
     ;oplot,[-100,100],[-100,100],thick=2,line=2,color=200
     ;window,3
     ;plot,ncat.r2,ccat.r2,ps=3,xr=[-2,5],yr=[-2,5],xtitle='r2 (no shear)',ytitle='r2 (with shear)'
     ;oplot,[-100,100],[-100,100],thick=2,line=2,color=200
     
  endfor
  psclose
  prepare_plots,/reset
  stop
  badfields = [93,97,150]
  for i = 0,nfiles-1 do begin
     ncat = mrdfits(files_none[i],1)
     ncat = ncat[select_for_analysis(ncat)]
     ccat = mrdfits(files_const[i],1)
     ccat = ccat[select_for_analysis(ccat)]
     match,ccat.id,ncat.id,cind,nind
     ccat = ccat[cind]
     ncat = ncat[nind]
; Look up the field properties.
     g1 = (truth_g1[where(truth_field_id eq fieldnumbers[i])])[0]
     g2 = (truth_g2[where(truth_field_id eq fieldnumbers[i])])[0]
     psf_g1 = (truth_psf_g1[where(truth_field_id eq fieldnumbers[i])])[0]
     psf_g2 = (truth_psf_g2[where(truth_field_id eq fieldnumbers[i])])[0]
     
     de1_dg1 = (ccat.g1 - ncat.g1)/g1
     de2_dg2 = (ccat.g2 - ncat.g2)/g2
     if total(i eq badfields) gt 0 then begin
        if n_elements(badTrueR1) eq 0 then begin
           badTrueR1 = de1_dg1
           badEmpR1  = ccat.r1
        endif else begin
           badTrueR1 = [badTrueR1, de1_dg1]
           badEmpR1 = [badEmpR1, ccat.r1]
        endelse
     endif else begin
        if abs(g1) gt 0.01 then begin
           if n_elements(goodTrueR1) eq 0 then begin
              goodTrueR1 = de1_dg1
              goodEmpR1 = ccat.r1
           endif else begin
              goodTrueR1 = [goodTrueR1, de1_dg1]
              goodEmpR1 = [goodEmpR1, ccat.r1]
           endelse
        endif
     endelse
  endfor

  plothist,asinh(goodTrueR1),bin=0.01,xgt,ygt,/noplot
  plothist,asinh(badTrueR1),bin=0.01,xbt,ybt,/noplot
  ygt = ygt/total(ygt)
  ybt = ybt/total(ybt)

  window,0
  plot,xgt,ygt,ps=10,/ylog,yr=[1e-5,.1],xtitle='arcsinh(R1)',ytitle='N(R1)',charsize=2.,xr=[-5,5]
  oplot,xbt,ybt,ps=10,color=200 
  legend,['asinh(R1) (diff, all)','asinh(R1) (diff, bad)'],line=[0,0],color=[-1,200],box=0,/top,/left,charsize=1.5

  window,1
  plot,xgt,ybt - interpol(ygt, xgt, xbt), xtitle='arcsinh(R1)', ytitle='N(asinh(R1), bad) - N(asinh(R1), good)',charsize=2.,$
       xr=[-6,6],/xstyle,yr=[-0.003,0.003]


  plothist,asinh(goodEmpR1),bin=0.01,xge,yge,/noplot
  plothist,asinh(badEmpR1),bin=0.01,xbe,ybe,/noplot
  yge = yge/total(yge)
  ybe = ybe/total(ybe)
  

  window,2
  plot,xge,yge,ps=10,/ylog,yr=[1e-5,.1],xtitle='arcsinh(R1)',ytitle='N(R1)',charsize=2.,xr=[-5,5]
  oplot,xbe,ybe,ps=10,color=200 
  legend,['asinh(R1) (metacal, all)','asinh(R1) (metacal, bad)'],line=[0,0],color=[-1,200],box=0,/top,/left,charsize=1.5

  window,3
  plot,xbe,ybe - interpol(yge, xge, xbe), xtitle='arcsinh(R1)', ytitle='N(asinh(R1), bad) - N(asinh(R1), good)',charsize=2.,$
       xr=[-6,6],/xstyle,yr=[-0.003,0.003]


stop
end
