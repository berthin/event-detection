img = imread('/tmp/gray.png');
img = resize(img, [115 115]);

dft = fft2(img);
dft_shift = fftshift(dft);

m = 57;

x = repmat(-m:m, 2*m+1, 1);
y = repmat((-m:m)', 1, 2*m+1);

n = 1;
w = 1;
d0 = 10;
D = (x .** 2 + y .** 2) .** 0.5;

h2 = 1 - (1 + w .* D ./ (D  - d0 ) .** 2 * n) .** -1;
mesh(h2);

imshow(abs(ifft2(fftshift(dft_shift  .* h2))))


function [h2] = draw_filter (n, w, d0)
  m = 115;
  x = repmat(-m:m, 2*m+1, 1);
  y = repmat((-m:m)', 1, 2*m+1);
  D = (x .** 2 + y .** 2) .** 0.5;
  h2 = 1 - (1 + w .* D ./ (D  - d0 ) .** 2 * n) .** -1;
endfunction


function [G, H] = apply_filter (img, h)
  img = resize(img, [115 115]);

  dft = fft2(img, 2*115+1, 2*115+1);
  dft_shift = fftshift(dft);

  H = dft_shift .* h;
  G = dft_shift + H;
  H = ifft2(ifftshift(H), 2*115+1, 2*115+1);
  G = ifft2(ifftshift(G), 2*115+1, 2*115+1);
  
  H = uint8(real(H(1:115, 1:115)));
  G = uint8(real(G(1:115, 1:115)));

  subplot(1,3,1);
  mesh(h);
  subplot(1,3,2);
  imshow(H, []);
  subplot(1,3,3);
  imshow(G, []);
endfunction
