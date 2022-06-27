clear;figure(1);clf;
load ../out/griddata.mat
load ../data/data41_benchmark.mat
iframe = 0;
for it = 0:20:nt
    load(['../out/step_'  num2str(it)  '.mat'])
    tl = tiledlayout(1,2,"TileSpacing","normal","Padding","compact");
    nexttile;imagesc([dx/2 lx-dx/2], [dy/2 ly-dy/2], C');axis image;set(gca,'YDir','normal');colorbar;title('Phase field (one-way p2g)');colormap(gca,flip(gray))
    hold on
    pX = pX(pA==true);
    pY = pY(pA==true);
    pC = pC(pA==true);
    pT = pT(pA==true);
    scatter(pX,pY,5,"m","filled");axis image;colorbar
    hold off
    nexttile;imagesc([dx/2 lx-dx/2], [dy/2 ly-dy/2], T');axis image;set(gca,'YDir','normal');colorbar;title('Temperature field (two-way p2g and g2p)');colormap(gca,jet)
    sgtitle(it)
    drawnow
%     exportgraphics(tl,sprintf('frame_%04d.png',iframe),'Resolution',300)
    iframe = iframe + 1;
end

% system('ffmpeg -r 10 -i frame_%%04d.png -c libx264 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white" -y video.mp4')