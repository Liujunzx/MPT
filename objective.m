function f = objective(u_r,u_t,N,sampled_scores_display_rgb,sampled_scores_display_t,x)
u_r = double(u_r);
u_t = double(u_t);
sampled_scores_display_rgb = double(sampled_scores_display_rgb);
sampled_scores_display_t = double(sampled_scores_display_t);

D_Fusion = x(1).*sampled_scores_display_rgb + x(2).*sampled_scores_display_t;
D_Fusion_a = (x(1).*sampled_scores_display_rgb + x(2).*sampled_scores_display_t)-((x(1)*u_r - x(2)*u_t)*ones(N^(1/2),N^(1/2))) ;

f =-((max(max(D_Fusion))-x(1)*u_r - x(2)*u_t)/sqrt(sum(sum(D_Fusion_a.^2))/N));




