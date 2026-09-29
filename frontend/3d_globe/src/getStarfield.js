import * as THREE from "three";

/**
 * Vast, full space background for the Mars globe.
 * Soft additive stars, spectral color, photo Milky Way skysphere,
 * parallax shells, diffraction spikes. No broken atmosphere / zodiacal fog.
 */

/** Fixed galactic-center direction along the ribbon (demo convention). */
const MW_CORE_ALONG = 0.55;
/** Dark dust-lane centers along the ribbon (Great Rift–style gaps). */
const MW_DUST_LANES = [-0.35, 0.15, 1.35, -1.55, 2.4];

function makeSoftCircleTexture(size = 256, soft = 0.35) {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  // Clear to transparent — avoid opaque corners that look square when magnified.
  ctx.clearRect(0, 0, size, size);
  const cx = size / 2;
  const cy = size / 2;
  const radius = size / 2;
  const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, radius);
  g.addColorStop(0, "rgba(255,255,255,1)");
  g.addColorStop(Math.min(0.18, soft * 0.45), "rgba(255,255,255,0.92)");
  g.addColorStop(0.38, "rgba(255,255,255,0.42)");
  g.addColorStop(0.62, "rgba(255,255,255,0.12)");
  g.addColorStop(0.85, "rgba(255,255,255,0.03)");
  g.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = g;
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.fill();
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.generateMipmaps = true;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.anisotropy = 4;
  tex.needsUpdate = true;
  return tex;
}

/**
 * Photo Milky Way skysphere — the approach used by most Three.js solar-system /
 * globe demos (equirect panorama on a BackSide sphere), not a procedural torus.
 * Texture: ESO/S. Brunier GigaGalaxy Zoom (CC BY 4.0) — credit required.
 * https://www.eso.org/public/images/eso0932a/
 * @returns {{ group: THREE.Group, skyMat: THREE.MeshBasicMaterial }}
 */
function createMilkyWaySky() {
  const group = new THREE.Group();
  group.name = "milkyWaySky";

  const map = new THREE.TextureLoader().load("./textures/milkyway.jpg");
  map.colorSpace = THREE.SRGBColorSpace;
  map.anisotropy = 8;
  map.generateMipmaps = true;
  map.minFilter = THREE.LinearMipmapLinearFilter;
  map.magFilter = THREE.LinearFilter;

  const skyMat = new THREE.MeshBasicMaterial({
    map,
    side: THREE.BackSide,
    depthWrite: false,
    // Must depth-test: transparent sky draws late; without this it covers Mars.
    depthTest: true,
    fog: false,
    toneMapped: false,
    transparent: true,
    opacity: 0.95,
    color: new THREE.Color(0xb8c0d0),
  });
  // Stay at optical infinity + far-plane depth so the globe always wins the z-test.
  skyMat.onBeforeCompile = (shader) => {
    shader.vertexShader = shader.vertexShader.replace(
      "#include <project_vertex>",
      `vec4 mvPosition = vec4( mat3(modelViewMatrix) * transformed, 1.0 );
       gl_Position = projectionMatrix * mvPosition;
       // Force far-plane depth (sky never occludes nearer objects like Mars).
       gl_Position.z = gl_Position.w;`
    );
  };
  skyMat.customProgramCacheKey = () => "mw-sky-infinity-far";

  const sky = new THREE.Mesh(new THREE.SphereGeometry(80, 64, 48), skyMat);
  sky.name = "milkyWaySphere";
  sky.rotation.z = 0.48;
  sky.rotation.y = -0.35;
  group.add(sky);

  group.renderOrder = -10;
  return { group, skyMat };
}

/** Soft elliptical glow for distant galaxies (Andromeda / Magellanic-style). */
function makeGalaxySmudgeTexture() {
  const size = 128;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, size, size);
  const cx = size / 2;
  const cy = size / 2;
  const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, size * 0.48);
  g.addColorStop(0, "rgba(220,230,255,0.55)");
  g.addColorStop(0.35, "rgba(180,200,255,0.22)");
  g.addColorStop(0.7, "rgba(140,160,220,0.06)");
  g.addColorStop(1, "rgba(100,120,180,0)");
  ctx.fillStyle = g;
  ctx.beginPath();
  ctx.ellipse(cx, cy, size * 0.46, size * 0.22, 0, 0, Math.PI * 2);
  ctx.fill();
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

/**
 * A few named deep-sky smudges on the celestial sphere (demo-scale, not to scale).
 * @returns {THREE.Group}
 */
function createDeepSkyObjects() {
  const group = new THREE.Group();
  group.name = "deepSkyObjects";
  const map = makeGalaxySmudgeTexture();

  const objects = [
    { name: "Andromeda", dir: new THREE.Vector3(0.55, 0.35, -0.75), scale: 2.8, color: 0xc8d4ff, opacity: 0.28 },
    { name: "LMC", dir: new THREE.Vector3(-0.65, -0.45, 0.55), scale: 2.2, color: 0xffe8d8, opacity: 0.32 },
    { name: "SMC", dir: new THREE.Vector3(-0.72, -0.38, 0.42), scale: 1.35, color: 0xffe0d0, opacity: 0.26 },
  ];

  for (const obj of objects) {
    const mat = new THREE.SpriteMaterial({
      map,
      color: obj.color,
      transparent: true,
      opacity: obj.opacity,
      depthWrite: false,
      depthTest: true,
      blending: THREE.AdditiveBlending,
      toneMapped: false,
    });
    const sprite = new THREE.Sprite(mat);
    sprite.name = obj.name;
    const p = obj.dir.clone().normalize().multiplyScalar(62);
    sprite.position.copy(p);
    sprite.scale.set(obj.scale * 1.6, obj.scale, 1);
    group.add(sprite);
  }
  group.renderOrder = -4;
  return group;
}

function makeDiffractionSpikeTexture(size = 256) {
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
    if (horizontal) ctx.fillRect(0, cy - 1.5, size, 3);
    else ctx.fillRect(cx - 1.5, 0, 3, size);
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
  tex.generateMipmaps = true;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.needsUpdate = true;
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
  /** Per-star GPU glitter (brightness twinkle). Skip for soft MW haze. */
  glitter = false,
  glitterAmount = 0.35,
  /** Power-law size scatter: many faint, few bright (magnitude-like). */
  magPower = 2.1,
  sizeMin = 0.45,
  sizeMax = 1.15,
}) {
  const verts = [];
  const colors = [];
  const seeds = [];
  const sizes = [];
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
    seeds.push(Math.random() * Math.PI * 2, 0.55 + Math.random() * 1.8, 0.45 + Math.random() * 0.55);
    // Skew small: most stars stay pinpricks; a few get larger discs.
    const mag = Math.pow(Math.random(), magPower);
    sizes.push(sizeMin + mag * (sizeMax - sizeMin));
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(verts, 3));
  geo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  geo.setAttribute("aSize", new THREE.Float32BufferAttribute(sizes, 1));
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

  mat.userData.hasSizeAttr = true;
  if (glitter) {
    mat.userData.glitter = true;
    mat.userData.baseGlitter = glitterAmount;
    mat.userData.uTime = { value: 0 };
    mat.userData.uGlitter = { value: glitterAmount };
  }
  mat.onBeforeCompile = (shader) => {
    if (glitter) {
      shader.uniforms.uTime = mat.userData.uTime;
      shader.uniforms.uGlitter = mat.userData.uGlitter;
    }
    shader.vertexShader = shader.vertexShader
      .replace(
        "#include <common>",
        `#include <common>
         attribute float aSize;
         ${glitter ? `attribute vec3 aTwinkle;
         uniform float uTime;
         uniform float uGlitter;
         varying float vTwinkle;` : ""}`
      )
      .replace(
        "#include <color_vertex>",
        glitter
          ? `#include <color_vertex>
           float tw = 0.5 + 0.5 * sin(uTime * aTwinkle.y + aTwinkle.x);
           tw *= 0.7 + 0.3 * sin(uTime * (aTwinkle.y * 1.6) + aTwinkle.x * 2.0);
           vTwinkle = mix(1.0, tw, uGlitter * aTwinkle.z);`
          : `#include <color_vertex>`
      )
      .replace(
        "#include <project_vertex>",
        glitter
          ? `#include <project_vertex>
           gl_PointSize *= aSize;
           // Brightness glitter — almost no size pulse (avoids zoom pixelation).
           gl_PointSize *= (0.97 + 0.05 * vTwinkle);
           gl_PointSize = min(gl_PointSize, 40.0);`
          : `#include <project_vertex>
           gl_PointSize *= aSize;
           gl_PointSize = min(gl_PointSize, 40.0);`
      );
    if (glitter) {
      shader.fragmentShader = shader.fragmentShader
        .replace(
          "#include <common>",
          `#include <common>
           varying float vTwinkle;`
        )
        .replace(
          "#include <opaque_fragment>",
          `#include <opaque_fragment>
           float sparkle = 0.78 + 0.32 * vTwinkle;
           gl_FragColor.rgb *= sparkle;
           gl_FragColor.a *= (0.72 + 0.28 * vTwinkle);`
        );
    }
  };
  mat.customProgramCacheKey = () =>
    glitter ? `star-mag-glitter-${glitterAmount}` : "star-mag";

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

  // High-res soft sprites + mipmaps so zoomed stars stay round, not blocky.
  const softStar = makeSoftCircleTexture(256, 0.42);
  const softBright = makeSoftCircleTexture(256, 0.38);
  const spikeMap = makeDiffractionSpikeTexture(256);

  const near = new THREE.Group();
  near.name = "starsNear";
  const mid = new THREE.Group();
  mid.name = "starsMid";
  const far = new THREE.Group();
  far.name = "starsFar";
  root.add(far, mid, near);

  // Photo MW skysphere — glitter point-stars sit on top for sparkle / depth.
  const { group: milkyWay, skyMat: mwSkyMat } = createMilkyWaySky();
  far.add(milkyWay);

  const deepSky = createDeepSkyObjects();
  far.add(deepSky);

  // Dense deep field — vastness (kept dense for dimensional depth)
  const farField = buildPoints(Math.floor(numStars * 0.4), {
    inBand: false,
    radiusMin: 46,
    radiusMax: 58,
    size: 0.18,
    map: softStar,
    opacity: 0.82,
    colorFn: () => stellarColor("field"),
    glitter: true,
    glitterAmount: 0.4,
    magPower: 2.4,
    sizeMin: 0.4,
    sizeMax: 1.05,
  });
  farField.renderOrder = -3;
  far.add(farField);

  const farBand = buildPoints(Math.floor(numStars * 0.28), {
    inBand: true,
    radiusMin: 46,
    radiusMax: 58,
    size: 0.22,
    map: softStar,
    opacity: 0.78,
    sampleFn: (r) => {
      const preferCore = Math.random() < 0.35;
      let s = milkyWaySample(r, { heightScale: 0.16, preferCore });
      if (inDustLane(s.along) && Math.random() < 0.7) {
        s = milkyWaySample(r, { heightScale: 0.16, preferCore: false });
      }
      return s;
    },
    colorFn: (_i, along) => {
      if (nearGalacticCore(along)) return stellarColor("bright");
      return stellarColor("band");
    },
    glitter: true,
    glitterAmount: 0.38,
    magPower: 2.2,
    sizeMin: 0.45,
    sizeMax: 1.2,
  });
  farBand.renderOrder = -2;
  far.add(farBand);

  // MID
  const midField = buildPoints(Math.floor(numStars * 0.38), {
    inBand: false,
    radiusMin: 34,
    radiusMax: 46,
    size: 0.24,
    map: softStar,
    opacity: 0.9,
    colorFn: () => stellarColor("field"),
    glitter: true,
    glitterAmount: 0.45,
    magPower: 2.2,
    sizeMin: 0.5,
    sizeMax: 1.15,
  });
  midField.renderOrder = -2;
  mid.add(midField);

  const midBand = buildPoints(Math.floor(numStars * 0.28), {
    inBand: true,
    radiusMin: 34,
    radiusMax: 46,
    size: 0.26,
    map: softStar,
    opacity: 0.85,
    colorFn: () => stellarColor("band"),
    glitter: true,
    glitterAmount: 0.42,
    magPower: 2.0,
    sizeMin: 0.5,
    sizeMax: 1.25,
  });
  midBand.renderOrder = -1;
  mid.add(midBand);

  const bright = buildPoints(Math.floor(numStars * 0.08), {
    inBand: false,
    radiusMin: 34,
    radiusMax: 46,
    size: 0.55,
    map: softBright,
    opacity: 0.96,
    colorFn: () => stellarColor("bright"),
    glitter: true,
    glitterAmount: 0.5,
    magPower: 1.6,
    sizeMin: 0.65,
    sizeMax: 1.35,
  });
  fillPositions(bright, 34, 46, 0.6);
  mid.add(bright);

  // NEAR — soft parallax punch (kept subtle so sky stays “fixed celestial”)
  const nearField = buildPoints(Math.floor(numStars * 0.12), {
    inBand: false,
    radiusMin: 24,
    radiusMax: 34,
    size: 0.28,
    map: softStar,
    opacity: 0.92,
    colorFn: () => stellarColor("field"),
    glitter: true,
    glitterAmount: 0.48,
    magPower: 2.0,
    sizeMin: 0.55,
    sizeMax: 1.2,
  });
  near.add(nearField);

  const nearBright = buildPoints(Math.floor(numStars * 0.05), {
    inBand: false,
    radiusMin: 24,
    radiusMax: 34,
    size: 0.42,
    map: softBright,
    opacity: 0.94,
    colorFn: () => stellarColor("bright"),
    glitter: true,
    glitterAmount: 0.52,
    magPower: 1.5,
    sizeMin: 0.7,
    sizeMax: 1.3,
  });
  fillPositions(nearBright, 24, 34, 0.45);
  near.add(nearBright);

  const hotCount = 18;
  const hot = buildPoints(hotCount, {
    inBand: false,
    radiusMin: 25,
    radiusMax: 36,
    size: 1.0,
    map: softBright,
    opacity: 0.92,
    colorFn: () => stellarColor("hot"),
    glitter: true,
    glitterAmount: 0.6,
    magPower: 1.2,
    sizeMin: 0.85,
    sizeMax: 1.25,
  });
  fillPositions(hot, 25, 36, 0.5);
  near.add(hot);

  // Diffraction spikes only on the few brightest (naked-eye / demo cue)
  const spikes = new THREE.Group();
  spikes.name = "diffractionSpikes";
  near.add(spikes);
  const hotPos = hot.geometry.attributes.position;
  const hotCol = hot.geometry.attributes.color;
  const spikeCount = Math.min(8, hotPos.count);
  for (let i = 0; i < spikeCount; i++) {
    const col = new THREE.Color(hotCol.getX(i), hotCol.getY(i), hotCol.getZ(i));
    const mat = new THREE.SpriteMaterial({
      map: spikeMap,
      color: col,
      transparent: true,
      opacity: 0.42,
      depthWrite: false,
      depthTest: true,
      blending: THREE.AdditiveBlending,
      toneMapped: false,
    });
    const spike = new THREE.Sprite(mat);
    spike.position.set(hotPos.getX(i), hotPos.getY(i), hotPos.getZ(i));
    const scale = 1.0 + Math.random() * 0.7;
    spike.scale.set(scale, scale, 1);
    spike.material.rotation = (Math.random() - 0.5) * 0.35;
    spikes.add(spike);
  }

  const glitterMats = [farField, farBand, midField, midBand, bright, nearField, nearBright, hot]
    .map((p) => p.material)
    .filter((m) => m.userData?.glitter);
  const spikeMats = spikes.children.map((s) => s.material);
  const starPointMats = [farField, farBand, midField, midBand, bright, nearField, nearBright, hot].map(
    (p) => p.material
  );
  for (const m of starPointMats) {
    m.userData.baseOpacity = m.opacity;
  }

  const settings = {
    exposureMode: "eye", // "cinematic" | "eye"
    mwBreathe: true,
    glitter: true,
    deepSky: true,
  };

  const cinematicMwColor = new THREE.Color(0xb8c0d0);
  const eyeMwColor = new THREE.Color(0x6a7388);
  const cinematicMwOpacity = 0.95;
  const eyeMwOpacity = 0.62;

  function applyExposureVisuals() {
    const eye = settings.exposureMode === "eye";
    mwSkyMat.color.copy(eye ? eyeMwColor : cinematicMwColor);
    const baseOp = eye ? eyeMwOpacity : cinematicMwOpacity;
    mwSkyMat.userData.baseOpacity = baseOp;
    mwSkyMat.opacity = baseOp;
    const opacityScale = eye ? 0.72 : 1;
    for (const m of starPointMats) {
      m.opacity = (m.userData.baseOpacity ?? 1) * opacityScale;
    }
    for (const m of spikeMats) {
      m.userData.baseSpikeOpacity = eye ? 0.18 : 0.3;
    }
    deepSky.visible = settings.deepSky;
    for (const sprite of deepSky.children) {
      if (sprite.material) {
        sprite.material.opacity = eye
          ? (sprite.userData.baseOpacity ?? 0.28) * 0.55
          : (sprite.userData.baseOpacity ?? 0.28);
      }
    }
  }

  for (const sprite of deepSky.children) {
    sprite.userData.baseOpacity = sprite.material.opacity;
  }

  function applyGlitterSetting() {
    for (const mat of glitterMats) {
      const base = mat.userData.baseGlitter ?? 0.4;
      const amount = settings.glitter ? base : 0;
      mat.userData.uGlitter.value = amount;
    }
  }

  function applySettings(partial = {}) {
    Object.assign(settings, partial);
    applyExposureVisuals();
    applyGlitterSetting();
    return { ...settings };
  }

  applySettings({});

  /**
   * @param {number} tSec
   */
  root.userData.update = (tSec) => {
    // Gentle shell drift — depth cue without a spinning sky
    near.rotation.y = tSec * 0.0035;
    near.rotation.x = Math.sin(tSec * 0.0007) * 0.014;
    mid.rotation.y = tSec * 0.0018;
    mid.rotation.x = Math.sin(tSec * 0.00045) * 0.008;
    far.rotation.y = tSec * 0.0008;
    far.rotation.x = Math.sin(tSec * 0.00025) * 0.004;

    if (settings.mwBreathe) {
      const base = mwSkyMat.userData.baseOpacity ?? cinematicMwOpacity;
      mwSkyMat.opacity = base * (1 + 0.05 * Math.sin(tSec * ((Math.PI * 2) / 30)));
    } else {
      mwSkyMat.opacity = mwSkyMat.userData.baseOpacity ?? cinematicMwOpacity;
    }

    for (const mat of glitterMats) {
      if (mat.userData.uTime) mat.userData.uTime.value = tSec;
    }
    for (let i = 0; i < spikeMats.length; i++) {
      const base = spikeMats[i].userData.baseSpikeOpacity ?? 0.3;
      spikeMats[i].opacity = settings.glitter
        ? base + Math.sin(tSec * 1.2 + i * 0.9) * 0.14
        : base * 0.5;
    }
  };

  root.userData.getSettings = () => ({ ...settings });
  root.userData.setSettings = applySettings;

  return root;
}
