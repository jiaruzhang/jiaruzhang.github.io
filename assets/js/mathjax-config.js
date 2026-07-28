// MathJax 配置（覆盖 Wowchemy 主题自带的同名文件）。
//
// 除了主题原有的设置，这里额外补了一批 MathJax 内核没有、但论文 LaTeX 里常用的宏。
// 这些命令在论文里来自各种符号包（txfonts、mathtools、amsmath 的 \DeclareMathOperator 等），
// MathJax 不认识；又因为主题启用了 noerrors 扩展，未定义命令不会报错，
// 而是把原始 TeX 直接当文本显示出来（例如页面上直接看到 "\nperp"），很难察觉。
//
// 新增宏时注意：只加 MathJax 本来就没有的命令，不要覆盖内置命令。
window.MathJax = {
  options: {
    // Don't render math in mindmaps as Markmap has its own math renderer.
    ignoreHtmlClass: 'markmap',
  },
  tex: {
    inlineMath: [
      ['$', '$'],
      ['\\(', '\\)'],
    ],
    displayMath: [
      ['$$', '$$'],
      ['\\[', '\\]'],
    ],
    processEscapes: false,
    packages: { '[+]': ['noerrors'] },
    macros: {
      // bm 宏包的粗体数学（MathJax 没有，论文里很常用）
      bm: ['\\boldsymbol{#1}', 1],
      // 条件独立 / 不独立（因果推断论文常用，来自 txfonts 等符号包）
      nperp: '\\not\\perp',
      indep: '\\perp\\!\\!\\!\\perp',
      nindep: '\\not\\perp\\!\\!\\!\\perp',
      // mathtools 的定义符号
      coloneqq: '\\mathrel{:=}',
      Coloneqq: '\\mathrel{::=}',
      eqqcolon: '\\mathrel{=:}',
      coloneq: '\\mathrel{:-}',
      // 常见的 \DeclareMathOperator
      argmin: '\\operatorname*{arg\\,min}',
      argmax: '\\operatorname*{arg\\,max}',
      Tr: '\\operatorname{Tr}',
      tr: '\\operatorname{tr}',
      sign: '\\operatorname{sign}',
      diag: '\\operatorname{diag}',
      rank: '\\operatorname{rank}',
      supp: '\\operatorname{supp}',
      KL: '\\operatorname{KL}',
      Var: '\\operatorname{Var}',
      Cov: '\\operatorname{Cov}',
      // 数集简写
      R: '\\mathbb{R}',
      N: '\\mathbb{N}',
      Z: '\\mathbb{Z}',
      Expect: '\\mathbb{E}',
    },
  },
  loader: {
    load: ['[tex]/noerrors'],
  },
};
