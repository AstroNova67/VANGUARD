import * as THREE from "three";

/**
 * Vast, full space background for the Mars globe.
 * Soft additive stars, spectral color, Milky Way density + soft band glow,
 * parallax shells, diffraction spikes. No broken atmosphere / zodiacal fog.
 */

function makeSoftCircleTexture(size = 64, soft = 0.4) {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  const g = ctx.createRadialGradient(
    size / 2,
    size / 2,
    0,
    size / 2,
    size / 2,
    size / 2
  );
  g.addColorStop(0, "rgba(255,255,255,1)");
  g.addColorStop(Math.min(0.35, soft), "rgba(255,255,255,0.9)");
  g.addColorStop(0.55, "rgba(255,255,255,0.28)");
  g.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.needsUpdate = true;
  return tex;
}

function makeDustTexture(size = 128) {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  const g = ctx.createRadialGradient(
    size / 2,
    size / 2,
    0,
    size / 2,
    size / 2,
    size / 2
  );
  g.addColorStop(0, "rgba(255,255,255,0.5)");
  g.addColorStop(0.35, "rgba(255,255,255,0.14)");
  g.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function makeDiffractionSpikeTexture(size = 128) {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, size, size);
  const cx = size / 2;
  const cy = size / 2;

  const drawArm = (horizontal) => {
    const grad = horizontal
      ? ctx.createLinearGradient(0, cy, size, cy)
      : ctx.createLinearGradient(cx, 0, cx, size);
    grad.addColorStop(0, "rgba(255,255,255,0)");
    grad.addColorStop(0.35, "rgba(255,255,255,0.18)");
    grad.addColorStop(0.5, "rgba(255,255,255,0.9)");
    grad.addColorStop(0.65, "rgba(255,255,255,0.18)");
    grad.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = grad;
    if (horizontal) ctx.fillRect(0, cy - 1.2, size, 2.4);
    else ctx.fillRect(cx - 1.2, 0, 2.4, size);
  };
  drawArm(true);
  drawArm(false);

  const core = ctx.createRadialGradient(cx, cy, 0, cx, cy, size * 0.08);
  core.addColorStop(0, "rgba(255,255,255,0.95)");
  core.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = core;
  ctx.beginPath();
  ctx.arc(cx, cy, size * 0.08, 0, Math.PI * 2);
  ctx.fill();

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function spherePoint(radius) {
  const u = Math.random();
  const v = Math.random();
  const theta = 2 * Math.PI * u;
  const phi = Math.acos(2 * v - 1);
  return new THREE.Vector3(
    radius * Math.sin(phi) * Math.cos(theta),
    radius * Math.sin(phi) * Math.sin(theta),
    radius * Math.cos(phi)
  );
}

function wrapPi(a) {
  let x = a;
  while (x > Math.PI) x -= Math.PI * 2;
  while (x < -Math.PI) x += Math.PI * 2;
  return x;
}

/** Fixed galactic-center direction along the ribbon (demo convention). */
const MW_CORE_ALONG = 0.55;
/** Dark dust-lane centers along the ribbon (Great Rift–style gaps). */
const MW_DUST_LANES = [-0.35, 0.15, 1.35, -1.55, 2.4];

function inDustLane(along) {
  for (const lane of MW_DUST_LANES) {
    if (Math.abs(wrapPi(along - lane)) < 0.16) return true;
  }
  return false;
}

function nearGalacticCore(along) {
  return Math.abs(wrapPi(along - MW_CORE_ALONG)) < 0.55;
}

/**
 * Sample a point on the galactic plane ribbon.
 * @returns {{ pos: THREE.Vector3, along: number }}
 */
function milkyWaySample(radius, { heightScale = 0.2, preferCore = false } = {}) {
  let along = (Math.random() - 0.5) * Math.PI * 2;
  if (preferCore) {
    along = MW_CORE_ALONG + (Math.random() - 0.5) * 1.1;
  }
  const height = (Math.random() - 0.5) * heightScale;
  const x = radius * Math.cos(along);
  const y = radius * height;
  const z = radius * Math.sin(along);
  const tilt = 0.48;
  return {
    along,
    pos: new THREE.Vector3(
      x,
      y * Math.cos(tilt) - z * Math.sin(tilt),
      y * Math.sin(tilt) + z * Math.cos(tilt)
    ),
  };
}

function milkyWayPoint(radius) {
  return milkyWaySample(radius).pos;
}

function stellarColor(tier = "field") {
  const roll = Math.random();
  let hue;
  let sat;
  let light;
  if (roll < 0.08) {
    hue = 0.58 + Math.random() * 0.06;
    sat = 0.7 + Math.random() * 0.25;
    light = 0.72 + Math.random() * 0.2;
  } else if (roll < 0.22) {
    hue = 0.55 + Math.random() * 0.05;
    sat = 0.45 + Math.random() * 0.25;
    light = 0.78 + Math.random() * 0.15;
  } else if (roll < 0.4) {
    hue = 0.12 + Math.random() * 0.08;
    sat = 0.2 + Math.random() * 0.2;
    light = 0.82 + Math.random() * 0.12;
  } else if (roll < 0.58) {
    hue = 0.11 + Math.random() * 0.04;
    sat = 0.55 + Math.random() * 0.3;
    light = 0.7 + Math.random() * 0.18;
  } else if (roll < 0.78) {
    hue = 0.06 + Math.random() * 0.04;
    sat = 0.65 + Math.random() * 0.28;
    light = 0.62 + Math.random() * 0.2;
  } else {
    hue = 0.02 + Math.random() * 0.03;
    sat = 0.72 + Math.random() * 0.25;
    light = 0.55 + Math.random() * 0.22;
  }

  if (tier === "band") {
    sat = Math.min(1, sat + 0.08);
    light = Math.min(0.95, light + 0.04);
  } else if (tier === "bright") {
    sat = Math.min(1, sat + 0.12);
    light = Math.min(0.98, light + 0.1);
  } else if (tier === "hot") {
    if (Math.random() < 0.55) {
      hue = 0.58 + Math.random() * 0.07;
      sat = 0.75 + Math.random() * 0.2;
      light = 0.8 + Math.random() * 0.15;
    } else {
      hue = 0.08 + Math.random() * 0.05;
      sat = 0.7 + Math.random() * 0.25;
      light = 0.75 + Math.random() * 0.15;
    }
  }

  return new THREE.Color().setHSL(hue, sat, light);
}

function buildPoints(count, {
  inBand = false,
  radiusMin = 28,
  radiusMax = 48,
  size = 0.25,
  map = null,
  opacity = 1,
  colorFn,
  depthTest = true,
  blending = THREE.AdditiveBlending,
  sampleFn = null,
  /** Per-star GPU glitter (size + brightness twinkle). Skip for soft MW haze. */
  glitter = false,
  glitterAmount = 0.35,
}) {
  const verts = [];
  const colors = [];
  const seeds = [];
  for (let i = 0; i < count; i++) {
    const r = radiusMin + Math.random() * (radiusMax - radiusMin);
    let pos;
    let along = 0;
    if (sampleFn) {
      const s = sampleFn(r);
      pos = s.pos;
      along = s.along;
    } else if (inBand) {
      const s = milkyWaySample(r);
      pos = s.pos;
      along = s.along;
    } else {
      pos = spherePoint(r);
    }
    verts.push(pos.x, pos.y, pos.z);
    const c = colorFn(i, along);
    colors.push(c.r, c.g, c.b);
    // seed.x = phase, seed.y = speed, seed.z = amplitude scale
    seeds.push(Math.random() * Math.PI * 2, 0.6 + Math.random() * 2.4, 0.5 + Math.random() * 0.5);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(verts, 3));
  geo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  if (glitter) {
    geo.setAttribute("aTwinkle", new THREE.Float32BufferAttribute(seeds, 3));
  }
  const mat = new THREE.PointsMaterial({
    size,
    map,
    vertexColors: true,
    transparent: true,
    opacity,
    depthWrite: false,
    depthTest,
    blending,
    sizeAttenuation: true,
    fog: false,
  });

  if (glitter) {
    mat.userData.glitter = true;
    mat.userData.uTime = { value: 0 };
    mat.userData.uGlitter = { value: glitterAmount };
    mat.onBeforeCompile = (shader) => {
      shader.uniforms.uTime = mat.userData.uTime;
      shader.uniforms.uGlitter = mat.userData.uGlitter;
      shader.vertexShader = shader.vertexShader
        .replace(
          "#include <common>",
          `#include <common>
           attribute vec3 aTwinkle;
           uniform float uTime;
           uniform float uGlitter;
           varying float vTwinkle;`
        )
        .replace(
          "#include <color_vertex>",
          `#include <color_vertex>
           float tw = 0.5 + 0.5 * sin(uTime * aTwinkle.y + aTwinkle.x);
           tw *= 0.65 + 0.35 * sin(uTime * (aTwinkle.y * 1.7) + aTwinkle.x * 2.1);
           vTwinkle = mix(1.0, tw, uGlitter * aTwinkle.z);`
        )
        .replace(
          "#include <project_vertex>",
          `#include <project_vertex>
           gl_PointSize *= (0.72 + 0.55 * vTwinkle);`
        );
      shader.fragmentShader = shader.fragmentShader
        .replace(
          "#include <common>",
          `#include <common>
           varying float vTwinkle;`
        )
        .replace(
          "#include <opaque_fragment>",
          `#include <opaque_fragment>
           gl_FragColor.rgb *= (0.55 + 0.75 * vTwinkle);
           gl_FragColor.a *= (0.5 + 0.65 * vTwinkle);`
        );
    };
    // Force shader rebuild when material is used
    mat.customProgramCacheKey = () => `star-glitter-${glitterAmount}`;
  }

  const points = new THREE.Points(geo, mat);
  points.frustumCulled = false;
  return points;
}

function fillPositions(points, radiusMin, radiusMax, bandFrac) {
  const pos = points.geometry.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    const r = radiusMin + Math.random() * (radiusMax - radiusMin);
    const p = Math.random() < bandFrac ? milkyWayPoint(r) : spherePoint(r);
    pos.setXYZ(i, p.x, p.y, p.z);
  }
  pos.needsUpdate = true;
}

/**
 * @param {{ numStars?: number }} [opts]
 * @returns {THREE.Group}
 */
export default function getStarfield({ numStars = 4200 } = {}) {
  const root = new THREE.Group();
  root.name = "starfield";

  const softStar = makeSoftCircleTexture(64, 0.4);
  const softBright = makeSoftCircleTexture(96, 0.35);
  const dustMap = makeDustTexture(128);
  const spikeMap = makeDiffractionSpikeTexture(128);

  const near = new THREE.Group();
  near.name = "starsNear";
  const mid = new THREE.Group();
  mid.name = "starsMid";
  const far = new THREE.Group();
  far.name = "starsFar";
  root.add(far, mid, near);

  // FAR — Milky Way ribbon haze (photo-like: bright lane + warmer core)
  // Skip most haze inside dust-lane angles so dark rifts can read.
  const dust = buildPoints(Math.floor(numStars * 0.24), {
    inBand: true,
    radiusMin: 46,
    radiusMax: 58,
    size: 3.6,
    map: dustMap,
    opacity: 0.3,
    sampleFn: (r) => {
      let s = milkyWaySample(r, { heightScale: 0.2 });
      let guard = 0;
      while (inDustLane(s.along) && guard++ < 8) {
        s = milkyWaySample(r, { heightScale: 0.2 });
      }
      return s;
    },
    colorFn: (_i, along) => {
      if (nearGalacticCore(along)) {
        // Warm cream / gold toward the galactic center (like long-exposure photos)
        return new THREE.Color().setHSL(0.1 + Math.random() * 0.04, 0.35 + Math.random() * 0.25, 0.48 + Math.random() * 0.22);
      }
      const roll = Math.random();
      if (roll < 0.5) {
        return new THREE.Color().setHSL(0.58 + Math.random() * 0.08, 0.45 + Math.random() * 0.2, 0.4 + Math.random() * 0.2);
      }
      if (roll < 0.8) {
        return new THREE.Color().setHSL(0.62 + Math.random() * 0.08, 0.3 + Math.random() * 0.15, 0.42 + Math.random() * 0.18);
      }
      return new THREE.Color().setHSL(0.55 + Math.random() * 0.05, 0.15 + Math.random() * 0.1, 0.55 + Math.random() * 0.2);
    },
  });
  dust.renderOrder = -5;
  far.add(dust);

  // Brighter core bulge haze
  const coreHaze = buildPoints(Math.floor(numStars * 0.12), {
    radiusMin: 47,
    radiusMax: 56,
    size: 4.2,
    map: dustMap,
    opacity: 0.32,
    sampleFn: (r) => milkyWaySample(r, { heightScale: 0.28, preferCore: true }),
    colorFn: () =>
      new THREE.Color().setHSL(0.09 + Math.random() * 0.05, 0.4 + Math.random() * 0.25, 0.5 + Math.random() * 0.22),
  });
  coreHaze.renderOrder = -5;
  far.add(coreHaze);

  // Cool secondary haze (still skipping dust lanes)
  const milkyHaze = buildPoints(Math.floor(numStars * 0.12), {
    radiusMin: 47,
    radiusMax: 56,
    size: 2.5,
    map: dustMap,
    opacity: 0.2,
    sampleFn: (r) => {
      let s = milkyWaySample(r, { heightScale: 0.16 });
      let guard = 0;
      while ((inDustLane(s.along) || nearGalacticCore(s.along)) && guard++ < 6) {
        s = milkyWaySample(r, { heightScale: 0.16 });
      }
      return s;
    },
    colorFn: () =>
      new THREE.Color().setHSL(0.57 + Math.random() * 0.07, 0.4 + Math.random() * 0.2, 0.42 + Math.random() * 0.18),
  });
  milkyHaze.renderOrder = -4;
  far.add(milkyHaze);

  // Dark dust lanes (Normal blending so they carve structure into the bright ribbon)
  const dustLanes = buildPoints(Math.floor(numStars * 0.1), {
    radiusMin: 45.5,
    radiusMax: 57,
    size: 2.8,
    map: dustMap,
    opacity: 0.55,
    blending: THREE.NormalBlending,
    sampleFn: (r) => {
      const lane = MW_DUST_LANES[Math.floor(Math.random() * MW_DUST_LANES.length)];
      const along = lane + (Math.random() - 0.5) * 0.28;
      const height = (Math.random() - 0.5) * 0.14;
      const x = r * Math.cos(along);
      const y = r * height;
      const z = r * Math.sin(along);
      const tilt = 0.48;
      return {
        along,
        pos: new THREE.Vector3(
          x,
          y * Math.cos(tilt) - z * Math.sin(tilt),
          y * Math.sin(tilt) + z * Math.cos(tilt)
        ),
      };
    },
    colorFn: () =>
      new THREE.Color().setHSL(0.06 + Math.random() * 0.03, 0.35 + Math.random() * 0.2, 0.04 + Math.random() * 0.06),
  });
  dustLanes.renderOrder = -3;
  far.add(dustLanes);

  // MID cool haze for parallax (lanes skipped)
  const midHaze = buildPoints(Math.floor(numStars * 0.1), {
    radiusMin: 36,
    radiusMax: 45,
    size: 2.6,
    map: dustMap,
    opacity: 0.14,
    sampleFn: (r) => {
      let s = milkyWaySample(r, { heightScale: 0.18 });
      let guard = 0;
      while (inDustLane(s.along) && guard++ < 6) {
        s = milkyWaySample(r, { heightScale: 0.18 });
      }
      return s;
    },
    colorFn: (_i, along) => {
      if (nearGalacticCore(along)) {
        return new THREE.Color().setHSL(0.1, 0.4, 0.45);
      }
      return new THREE.Color().setHSL(0.58 + Math.random() * 0.08, 0.4 + Math.random() * 0.15, 0.38 + Math.random() * 0.15);
    },
  });
  midHaze.renderOrder = -3;
  mid.add(midHaze);

  // Dense deep field — vastness
  const farField = buildPoints(Math.floor(numStars * 0.4), {
    inBand: false,
    radiusMin: 46,
    radiusMax: 58,
    size: 0.2,
    map: softStar,
    opacity: 0.85,
    colorFn: () => stellarColor("field"),
    glitter: true,
    glitterAmount: 0.45,
  });
  farField.renderOrder = -3;
  far.add(farField);

  const farBand = buildPoints(Math.floor(numStars * 0.32), {
    inBand: true,
    radiusMin: 46,
    radiusMax: 58,
    size: 0.26,
    map: softStar,
    opacity: 0.95,
    sampleFn: (r) => {
      // Pack more stars toward the core; thin them in dust lanes
      const preferCore = Math.random() < 0.35;
      let s = milkyWaySample(r, { heightScale: 0.18, preferCore });
      if (inDustLane(s.along) && Math.random() < 0.7) {
        s = milkyWaySample(r, { heightScale: 0.18, preferCore: false });
      }
      return s;
    },
    colorFn: (_i, along) => {
      if (nearGalacticCore(along)) return stellarColor("bright");
      return stellarColor("band");
    },
    glitter: true,
    glitterAmount: 0.4,
  });
  farBand.renderOrder = -2;
  far.add(farBand);

  // MID
  const midField = buildPoints(Math.floor(numStars * 0.38), {
    inBand: false,
    radiusMin: 34,
    radiusMax: 46,
    size: 0.26,
    map: softStar,
    opacity: 0.92,
    colorFn: () => stellarColor("field"),
    glitter: true,
    glitterAmount: 0.5,
  });
  midField.renderOrder = -2;
  mid.add(midField);

  const midBand = buildPoints(Math.floor(numStars * 0.4), {
    inBand: true,
    radiusMin: 34,
    radiusMax: 46,
    size: 0.3,
    map: softStar,
    opacity: 0.96,
    colorFn: () => stellarColor("band"),
    glitter: true,
    glitterAmount: 0.45,
  });
  midBand.renderOrder = -1;
  mid.add(midBand);

  const bright = buildPoints(Math.floor(numStars * 0.08), {
    inBand: false,
    radiusMin: 34,
    radiusMax: 46,
    size: 0.78,
    map: softBright,
    opacity: 0.98,
    colorFn: () => stellarColor("bright"),
    glitter: true,
    glitterAmount: 0.65,
  });
  fillPositions(bright, 34, 46, 0.6);
  mid.add(bright);

  // NEAR — parallax punch
  const nearField = buildPoints(Math.floor(numStars * 0.12), {
    inBand: false,
    radiusMin: 24,
    radiusMax: 34,
    size: 0.32,
    map: softStar,
    opacity: 0.94,
    colorFn: () => stellarColor("field"),
    glitter: true,
    glitterAmount: 0.55,
  });
  near.add(nearField);

  const nearBright = buildPoints(Math.floor(numStars * 0.05), {
    inBand: false,
    radiusMin: 24,
    radiusMax: 34,
    size: 0.55,
    map: softStar,
    opacity: 0.95,
    colorFn: () => stellarColor("bright"),
    glitter: true,
    glitterAmount: 0.7,
  });
  fillPositions(nearBright, 24, 34, 0.45);
  near.add(nearBright);

  const hotCount = 22;
  const hot = buildPoints(hotCount, {
    inBand: false,
    radiusMin: 25,
    radiusMax: 36,
    size: 1.7,
    map: softBright,
    opacity: 0.88,
    colorFn: () => stellarColor("hot"),
    glitter: true,
    glitterAmount: 0.85,
  });
  fillPositions(hot, 25, 36, 0.5);
  near.add(hot);

  const spikes = new THREE.Group();
  spikes.name = "diffractionSpikes";
  near.add(spikes);
  const hotPos = hot.geometry.attributes.position;
  const hotCol = hot.geometry.attributes.color;
  for (let i = 0; i < hotPos.count; i++) {
    const col = new THREE.Color(hotCol.getX(i), hotCol.getY(i), hotCol.getZ(i));
    const mat = new THREE.SpriteMaterial({
      map: spikeMap,
      color: col,
      transparent: true,
      opacity: 0.5,
      depthWrite: false,
      depthTest: true,
      blending: THREE.AdditiveBlending,
      toneMapped: false,
    });
    const spike = new THREE.Sprite(mat);
    spike.position.set(hotPos.getX(i), hotPos.getY(i), hotPos.getZ(i));
    const scale = 1.7 + Math.random() * 1.3;
    spike.scale.set(scale, scale, 1);
    spike.material.rotation = (Math.random() - 0.5) * 0.35;
    spikes.add(spike);
  }

  const glitterMats = [farField, farBand, midField, midBand, bright, nearField, nearBright, hot]
    .map((p) => p.material)
    .filter((m) => m.userData?.glitter);
  const spikeMats = spikes.children.map((s) => s.material);

  root.userData.update = (tSec) => {
    near.rotation.y = tSec * 0.007;
    near.rotation.x = Math.sin(tSec * 0.0011) * 0.03;
    mid.rotation.y = tSec * 0.0035;
    mid.rotation.x = Math.sin(tSec * 0.0007) * 0.018;
    far.rotation.y = tSec * 0.0016;
    far.rotation.x = Math.sin(tSec * 0.0004) * 0.01;

    for (const mat of glitterMats) {
      if (mat.userData.uTime) mat.userData.uTime.value = tSec;
    }
    for (let i = 0; i < spikeMats.length; i++) {
      spikeMats[i].opacity = 0.35 + Math.sin(tSec * 1.4 + i * 0.9) * 0.18;
    }
  };

  return root;
}
