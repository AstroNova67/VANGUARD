import * as THREE from "three";
import { GLTFLoader } from "jsm/loaders/GLTFLoader.js";

/**
 * Real Mars-system features for the globe scene.
 *
 * Moons: NASA/JPL VTAD glTF shapes + textures (Phobos / Deimos), orbital distances
 * track Mars radii (Phobos ~2.76 Rm, Deimos ~6.9 Rm). Physical sizes are exaggerated
 * ~15× so they read in a demo — true mean radii are ~11 km / ~6 km vs Mars ~3390 km.
 *
 * Occasional meteors, distant Earth point, moon sun catch-light.
 */

const MARS_RADIUS_KM = 3389.5;
const MARS_MESH_RADIUS = 2;

const PHOBOS_MODEL_URL = "./models/phobos.glb";
const DEIMOS_MODEL_URL = "./models/deimos.glb";

function kmToScene(km) {
  return (km / MARS_RADIUS_KM) * MARS_MESH_RADIUS;
}

function makeRockTexture(seed = 1) {
  const size = 128;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = seed === 1 ? "#5a4a3c" : "#6a5a4e";
  ctx.fillRect(0, 0, size, size);
  let s = seed * 99991;
  const rand = () => {
    s = (s * 16807) % 2147483647;
    return (s - 1) / 2147483646;
  };
  for (let i = 0; i < 900; i++) {
    const x = rand() * size;
    const y = rand() * size;
    const r = 0.4 + rand() * 2.8;
    const shade = 40 + Math.floor(rand() * 90);
    ctx.fillStyle = `rgba(${shade},${shade - 8},${shade - 16},${0.15 + rand() * 0.45})`;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }
  if (seed === 1) {
    ctx.fillStyle = "rgba(20,16,12,0.55)";
    ctx.beginPath();
    ctx.arc(size * 0.38, size * 0.42, size * 0.18, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "rgba(90,80,70,0.35)";
    ctx.lineWidth = 2;
    ctx.stroke();
  }
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  return tex;
}

/** Fallback potato mesh if NASA glTF fails to load. */
function makePotatoGeometry(radius, { squash = 1, stretch = 1.2, crater = false } = {}) {
  const geo = new THREE.IcosahedronGeometry(radius, 3);
  const pos = geo.attributes.position;
  const v = new THREE.Vector3();
  for (let i = 0; i < pos.count; i++) {
    v.fromBufferAttribute(pos, i);
    const n = v.clone().normalize();
    const noise =
      0.82 +
      0.18 * Math.sin(n.x * 6.1 + n.y * 4.3) * Math.cos(n.z * 5.7) +
      0.08 * Math.sin(n.x * 13.0 + n.z * 9.0);
    v.multiplyScalar(noise);
    v.x *= stretch;
    v.y *= squash;
    v.z *= 0.9;
    if (crater) {
      const toward = Math.max(0, n.x);
      v.x -= toward * radius * 0.22;
    }
    pos.setXYZ(i, v.x, v.y, v.z);
  }
  geo.computeVertexNormals();
  return geo;
}

function makeFallbackMoonMesh(name, { radius, color, seed, crater }) {
  const mat = new THREE.MeshStandardMaterial({
    map: makeRockTexture(seed),
    color,
    roughness: 0.72,
    metalness: 0.12,
    flatShading: true,
    emissive: new THREE.Color(0xffe0c0),
    emissiveIntensity: 0.04,
  });
  const mesh = new THREE.Mesh(
    makePotatoGeometry(radius, {
      stretch: name.startsWith("Phobos") ? 1.25 : 1.15,
      squash: name.startsWith("Phobos") ? 0.88 : 0.92,
      crater,
    }),
    mat
  );
  mesh.name = name;
  return mesh;
}

/** Tune NASA materials for sunlight + optional catch-light emissive. */
function prepareMoonMaterials(root) {
  root.traverse((obj) => {
    if (!obj.isMesh || !obj.material) return;
    const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
    for (const m of mats) {
      if (m.map) m.map.colorSpace = THREE.SRGBColorSpace;
      if ("roughness" in m) m.roughness = Math.min(0.92, m.roughness ?? 0.85);
      if ("metalness" in m) m.metalness = Math.min(0.08, m.metalness ?? 0.04);
      if ("emissive" in m) {
        m.emissive = new THREE.Color(0xffe0c0);
        m.emissiveIntensity = 0.03;
      }
      m.needsUpdate = true;
    }
  });
}

/**
 * Scale + center a loaded glTF so its longest axis matches targetRadius * 2.
 * @returns {THREE.Group}
 */
function fitMoonModel(gltfScene, targetRadius) {
  const wrapper = new THREE.Group();
  const model = gltfScene;
  wrapper.add(model);

  model.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(model);
  const center = new THREE.Vector3();
  const size = new THREE.Vector3();
  box.getCenter(center);
  box.getSize(size);
  model.position.sub(center);

  const maxDim = Math.max(size.x, size.y, size.z, 1e-6);
  wrapper.scale.setScalar((targetRadius * 2) / maxDim);

  prepareMoonMaterials(wrapper);
  return wrapper;
}

function forEachMoonMaterial(root, fn) {
  root.traverse((obj) => {
    if (!obj.isMesh || !obj.material) return;
    const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
    for (const m of mats) fn(m);
  });
}

const _moonLitDir = new THREE.Vector3();

/**
 * @param {THREE.Scene} scene
 * @returns {{
 *   update: (dt: number, sunWorldDir?: THREE.Vector3) => void,
 *   phobos: THREE.Object3D,
 *   deimos: THREE.Object3D,
 *   getSettings: () => object,
 *   setSettings: (partial: object) => object
 * }}
 */
export function addMarsMoons(scene) {
  const SIZE_EXAGGERATION = 15;

  const phobosOrbitR = kmToScene(9376);
  const deimosOrbitR = kmToScene(23460);
  const phobosRadius = kmToScene(11.08) * SIZE_EXAGGERATION;
  const deimosRadius = kmToScene(6.2) * SIZE_EXAGGERATION;

  const phobosPivot = new THREE.Object3D();
  phobosPivot.name = "phobosOrbit";
  phobosPivot.rotation.x = THREE.MathUtils.degToRad(1.1);
  scene.add(phobosPivot);

  const phobos = new THREE.Object3D();
  phobos.name = "Phobos";
  phobos.position.set(phobosOrbitR, 0, 0);
  phobosPivot.add(phobos);

  const phobosFallback = makeFallbackMoonMesh("PhobosFallback", {
    radius: phobosRadius,
    color: 0xb8a090,
    seed: 1,
    crater: true,
  });
  phobos.add(phobosFallback);

  const deimosPivot = new THREE.Object3D();
  deimosPivot.name = "deimosOrbit";
  deimosPivot.rotation.x = THREE.MathUtils.degToRad(1.8);
  deimosPivot.rotation.z = THREE.MathUtils.degToRad(8);
  scene.add(deimosPivot);

  const deimos = new THREE.Object3D();
  deimos.name = "Deimos";
  deimos.position.set(deimosOrbitR, 0, 0);
  deimosPivot.add(deimos);

  const deimosFallback = makeFallbackMoonMesh("DeimosFallback", {
    radius: deimosRadius,
    color: 0xc4b4a4,
    seed: 2,
    crater: false,
  });
  deimos.add(deimosFallback);

  const loader = new GLTFLoader();

  function loadMoonModel(url, anchor, fallback, targetRadius, label) {
    loader.load(
      url,
      (gltf) => {
        const fitted = fitMoonModel(gltf.scene, targetRadius);
        fitted.name = `${label}Model`;
        if (fallback.parent) fallback.parent.remove(fallback);
        fallback.geometry?.dispose?.();
        if (fallback.material) {
          fallback.material.map?.dispose?.();
          fallback.material.dispose?.();
        }
        anchor.add(fitted);
      },
      undefined,
      (err) => {
        console.warn(`[moons] Failed to load ${label} model; keeping procedural mesh.`, err);
      }
    );
  }

  loadMoonModel(PHOBOS_MODEL_URL, phobos, phobosFallback, phobosRadius, "Phobos");
  loadMoonModel(DEIMOS_MODEL_URL, deimos, deimosFallback, deimosRadius, "Deimos");

  const deimosOmega = 0.08;
  const phobosOmega = deimosOmega * (1.2625 / 0.3189);
  const marsCenter = new THREE.Vector3(0, 0, 0);
  const moonWorld = new THREE.Vector3();
  const toSun = new THREE.Vector3();

  const settings = {
    moonCatchLight: true,
  };

  function applyCatchLight(moon, sunWorldDir) {
    if (!settings.moonCatchLight || !sunWorldDir) {
      forEachMoonMaterial(moon, (m) => {
        if ("emissiveIntensity" in m) m.emissiveIntensity = 0.02;
        if ("roughness" in m) m.roughness = 0.9;
        if ("metalness" in m) m.metalness = 0.02;
      });
      return;
    }
    moon.getWorldPosition(moonWorld);
    toSun.copy(sunWorldDir).normalize();
    _moonLitDir.copy(moonWorld).normalize();
    const lit = Math.max(0, _moonLitDir.dot(toSun));
    forEachMoonMaterial(moon, (m) => {
      if ("emissiveIntensity" in m) m.emissiveIntensity = 0.03 + lit * 0.16;
      if ("roughness" in m) m.roughness = 0.78;
      if ("metalness" in m) m.metalness = 0.06;
    });
  }

  function applySettings(partial = {}) {
    Object.assign(settings, partial);
    return { ...settings };
  }

  return {
    phobos,
    deimos,
    getSettings: () => ({ ...settings }),
    setSettings: applySettings,
    update(dt, sunWorldDir) {
      phobosPivot.rotation.y += phobosOmega * dt;
      deimosPivot.rotation.y += deimosOmega * dt;
      phobos.lookAt(marsCenter);
      deimos.lookAt(marsCenter);
      applyCatchLight(phobos, sunWorldDir);
      applyCatchLight(deimos, sunWorldDir);
    },
  };
}

function makeEarthPointTexture() {
  const size = 64;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, size, size);
  const cx = size / 2;
  const cy = size / 2;
  const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, size / 2);
  g.addColorStop(0, "rgba(230,245,255,1)");
  g.addColorStop(0.25, "rgba(120,180,255,0.85)");
  g.addColorStop(0.55, "rgba(70,140,255,0.35)");
  g.addColorStop(1, "rgba(40,100,220,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

/**
 * Distant Earth as a tiny blue-white point near the sunward ecliptic (from Mars).
 * @param {THREE.Object3D} sunPivot
 */
export function addEarthFromMars(sunPivot) {
  const group = new THREE.Group();
  group.name = "earthFromMars";

  const mat = new THREE.SpriteMaterial({
    map: makeEarthPointTexture(),
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    opacity: 0.95,
    toneMapped: false,
  });
  const sprite = new THREE.Sprite(mat);
  sprite.name = "earthPoint";
  const dist = 52;
  const sep = 0.42;
  sprite.position.set(Math.cos(sep) * dist, 0.08, Math.sin(sep) * dist);
  sprite.scale.set(0.55, 0.55, 1);
  group.add(sprite);
  sunPivot.add(group);

  const settings = { earth: true };

  function applySettings(partial = {}) {
    Object.assign(settings, partial);
    group.visible = settings.earth;
    return { ...settings };
  }

  applySettings({});

  return {
    group,
    getSettings: () => ({ ...settings }),
    setSettings: applySettings,
    update(tSec) {
      if (!settings.earth) return;
      mat.opacity = 0.75 + 0.2 * Math.sin(tSec * 2.1);
      const s = 0.5 + 0.08 * Math.sin(tSec * 1.7 + 1.2);
      sprite.scale.set(s, s, 1);
    },
  };
}

/**
 * Occasional meteors — mostly faint; rare brighter fireballs.
 */
export function addMeteorShowers(scene) {
  const group = new THREE.Group();
  group.name = "meteors";
  scene.add(group);

  const pool = [];
  const POOL = 6;
  const geo = new THREE.BufferGeometry();
  geo.setAttribute(
    "position",
    new THREE.Float32BufferAttribute([0, 0, 0, 1, 0, 0], 3)
  );

  for (let i = 0; i < POOL; i++) {
    const mat = new THREE.LineBasicMaterial({
      color: 0xffe8d0,
      transparent: true,
      opacity: 0,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      toneMapped: false,
    });
    const line = new THREE.Line(geo, mat);
    line.visible = false;
    line.frustumCulled = false;
    group.add(line);
    pool.push({
      line,
      life: 0,
      maxLife: 0,
      speed: 0,
      dir: new THREE.Vector3(),
      origin: new THREE.Vector3(),
      length: 1,
      peakOpacity: 0.35,
      bright: false,
    });
  }

  const settings = {
    meteors: true,
    meteorRate: "rare",
  };

  let cooldown = 4 + Math.random() * 6;
  let brightCooldown = 25 + Math.random() * 40;
  const xAxis = new THREE.Vector3(1, 0, 0);
  const radial = new THREE.Vector3();
  const tangential = new THREE.Vector3();
  const tmp = new THREE.Vector3();

  function spherePointOnRadius(radius) {
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

  function spawn(m, bright) {
    const r = 30 + Math.random() * 14;
    const start = spherePointOnRadius(r);
    tangential
      .set(Math.random() - 0.5, (Math.random() - 0.5) * 0.7, Math.random() - 0.5)
      .normalize();
    radial.copy(start).normalize();
    tangential.addScaledVector(radial, -tangential.dot(radial)).normalize();
    if (tangential.lengthSq() < 0.2) {
      tangential.set(-radial.z, 0, radial.x).normalize();
    }
    m.origin.copy(start);
    m.dir.copy(tangential);
    m.bright = bright;
    m.length = bright ? 3.2 + Math.random() * 2.5 : 0.9 + Math.random() * 1.6;
    m.life = 0;
    m.maxLife = bright ? 0.7 + Math.random() * 0.45 : 0.35 + Math.random() * 0.4;
    m.speed = bright ? 28 + Math.random() * 18 : 16 + Math.random() * 16;
    m.peakOpacity = bright ? 0.95 : 0.22 + Math.random() * 0.18;
    m.line.visible = true;
    if (bright) {
      m.line.material.color.setHSL(0.08 + Math.random() * 0.06, 0.55, 0.92);
    } else {
      m.line.material.color.setHSL(0.1 + Math.random() * 0.08, 0.25, 0.8);
    }
    m.line.material.opacity = 0;
  }

  function nextFaintCooldown() {
    if (settings.meteorRate === "demo") return 1.8 + Math.random() * 3.5;
    return 6 + Math.random() * 10;
  }

  function nextBrightCooldown() {
    if (settings.meteorRate === "demo") return 12 + Math.random() * 18;
    return 35 + Math.random() * 50;
  }

  function applySettings(partial = {}) {
    Object.assign(settings, partial);
    group.visible = settings.meteors;
    return { ...settings };
  }

  return {
    getSettings: () => ({ ...settings }),
    setSettings: applySettings,
    update(dt) {
      if (!settings.meteors) {
        for (const m of pool) {
          m.line.visible = false;
          m.line.material.opacity = 0;
        }
        return;
      }

      cooldown -= dt;
      brightCooldown -= dt;

      if (brightCooldown <= 0) {
        const idle = pool.find((m) => !m.line.visible);
        if (idle) spawn(idle, true);
        brightCooldown = nextBrightCooldown();
      } else if (cooldown <= 0) {
        const idle = pool.find((m) => !m.line.visible);
        if (idle) spawn(idle, false);
        cooldown = nextFaintCooldown();
      }

      for (const m of pool) {
        if (!m.line.visible) continue;
        m.life += dt;
        const t = m.life / m.maxLife;
        if (t >= 1) {
          m.line.visible = false;
          m.line.material.opacity = 0;
          continue;
        }
        const fade = t < 0.15 ? t / 0.15 : t > 0.6 ? (1 - t) / 0.4 : 1;
        m.line.material.opacity = m.peakOpacity * fade;
        tmp.copy(m.origin).addScaledVector(m.dir, m.speed * m.life);
        m.line.position.copy(tmp);
        m.line.quaternion.setFromUnitVectors(xAxis, m.dir);
        m.line.scale.set(m.length, 1, 1);
      }
    },
  };
}
