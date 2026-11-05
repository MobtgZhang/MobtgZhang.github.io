---
# Processed by Jekyll so the sprite paths respect `site.baseurl`.
---

// ✰ webmeji ✰
// Site configuration for MobtgZhang.github.io
//
// Adapted from https://github.com/lars-rooij/webmeji (Unlicense, public domain).
// Only the bottom edge is enabled so the character stays on the lower-right
// corner of the page instead of climbing over the content.
//
// Frames live in /assets/webmeji/miku/ and are served from the site's own
// origin, so the character keeps working offline via the PWA service worker.

{% assign sprite = '/assets/webmeji/miku' | relative_url %}

window.SPAWNING = [
  { id: 'webmeji-0', config: 'SITE_MIKU_CONFIG' }
];

window.SITE_MIKU_CONFIG = {
  // pet   = hover animation
  // drag  = click or touch to pick up
  // bottom must always be enabled; top/left/right are intentionally omitted
  // so the character never leaves the lower-right corner.
  ALLOWANCES: ['pet', 'drag', 'bottom'],

  // Fraction of the viewport width the character may roam across, measured
  // from the right edge. 0.22 keeps it near the bottom-right corner.
  ZONE_RATIO: 0.22,

  // Height (px) reserved below the character so it never sits on top of the
  // site footer text or the "back to top" button.
  BOTTOM_OFFSET: 56,

  // Extra right-hand inset (px) on top of the auto-detected back-to-top
  // button, to keep a comfortable gap between them.
  RIGHT_INSET: 0,

  // movement and physics (pixels per frame)
  walkspeed: 32,
  fallspeed: 150,
  jumpspeed: 200,

  // time in ms before standing back up after falling
  gettingupspeed: 2600,

  // common idle and movement animations on the bottom edge
  walk: {
    frames: ["{{ sprite }}/shime1.png", "{{ sprite }}/shime2.png", "{{ sprite }}/shime3.png", "{{ sprite }}/shime2.png"],
    interval: 175, loops: 6},

  stand: {
    frames: ["{{ sprite }}/shime1.png"],
    interval: 1000, loops: 1},

  sit: {
    frames: ["{{ sprite }}/shime11.png"],
    interval: 1000, loops: 1,
    randomizeDuration: true, min: 3000, max: 9000},

  spin: {
    frames: ["{{ sprite }}/shime1.png"],
    interval: 150, loops: 3},

  dance: {
    frames: ["{{ sprite }}/shime5.png", "{{ sprite }}/shime6.png", "{{ sprite }}/shime1.png"],
    interval: 200, loops: 2},

  trip: {
    frames: ["{{ sprite }}/shime18.png", "{{ sprite }}/shime19.png", "{{ sprite }}/shime19.png"],
    interval: 250, loops: 1},

  // behavior flow control
  forcewalk: { // uses the walking frames
    loops: 6},

  forcethink: {
    frames: ["{{ sprite }}/shime27.png", "{{ sprite }}/shime28.png"],
    interval: 500, loops: 2},

  // user interaction animations
  pet: {
    frames: ["{{ sprite }}/shime15.png", "{{ sprite }}/shime16.png", "{{ sprite }}/shime17.png"],
    interval: 400},

  drag: {
    frames: ["{{ sprite }}/shime7.png", "{{ sprite }}/shime5.png", "{{ sprite }}/shime8.png", "{{ sprite }}/shime6.png"],
    interval: 210},

  // falling and recovery animations
  falling: {
    frames: ["{{ sprite }}/shime10.png", "{{ sprite }}/shime18.png"],
    interval: 200, loops: 2},

  fallen: {
    frames: ["{{ sprite }}/shime9.png", "{{ sprite }}/shime4.png", "{{ sprite }}/shime19.png"],
    interval: 250, loops: 1},

  // action frequency: an action listed more often happens more often
  ORIGINAL_ACTIONS: [
    'walk','walk','walk','walk','walk','walk',
    'spin','spin','spin',
    'sit','sit',
    'dance','dance','dance',
    'trip'
  ],

  EDGE_ACTIONS: [
    'hang'
  ],

  // probability of jumping to an edge; 0 disables it entirely
  JUMP_CHANCE: 0,

  // edge-specific animations (unused while only 'bottom' is allowed)
  climbSide: {
    frames: ["{{ sprite }}/shime13.png", "{{ sprite }}/shime14.png"],
    interval: 200, loops: 2},

  hangstillSide: {
    frames: ["{{ sprite }}/shime12.png"],
    interval: 200, loops: 2,
    randomizeDuration: true, min: 3000, max: 9000},

  climbTop: {
    frames: ["{{ sprite }}/shime24.png", "{{ sprite }}/shime25.png"],
    interval: 200, loops: 8},

  hangstillTop: {
    frames: ["{{ sprite }}/shime23.png"],
    interval: 200, loops: 2,
    randomizeDuration: true, min: 3000, max: 9000},

  jump: {
    frames: ["{{ sprite }}/shime22.png"],
    interval: 200}
};
