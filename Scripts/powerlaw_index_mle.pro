function p_iter_root,p
  common POWERLAW_INDEX_MLE_BLOCK, logavg, kmin, kmax
  root = (1-p) / logavg / kmin * (1 - (kmin/kmax)^p)/( 1 - (kmin/kmax)^(p-1))  + 1
  return,root
end

function powerlaw_index_mle,data, kmin, kmax,doplot=doplot,perr = perr
if n_elements(kmax) eq 0 then begin
   ndata = n_elements(data)*1.
   p = 2 - ndata / kmin /total(alog(data),/double)
   return,p
endif else begin
   common POWERLAW_INDEX_MLE_BLOCK, logavg, kmin_in, kmax_in
   logavg = mean(alog(data))
   kmin_in = kmin
   kmax_in = kmax
   ndata = n_elements(data)*1.
   ; Use for an initial guess what we'd have chosen for the unbounded case.
   pstart = 2 - ndata / kmin / total(alog(data),/double)
   pbest = fx_root([1.01,pstart,2*pstart],"p_iter_root")
   return,pbest
endelse
end


