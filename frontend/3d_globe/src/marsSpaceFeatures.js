import * as THREE from "three";

/**
 * Real Mars-system features for the globe scene.
 *
 * Moons: orbital distances track Mars radii (Phobos ~2.76 Rm, Deimos ~6.9 Rm).
 * Physical sizes are exaggerated ~15× so they read in a demo — true mean radii
 * are ~11 km / ~6 km vs Mars ~3390 km (invisible at 1:1 on this globe).
 *
 * Occasional meteors for sky life. No atmosphere wash / zodiacal fog overlays.
 */

const MARS_RADIUS_KM = 3389.5;
const MARS_MESH_RADIUS = 2;

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
  // Stickney-scale crater hint on Phobos (seed 1).
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

/** Irregular potato mesh — Phobos/Deimos are not spheres. */
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
      // Flatten a Stickney-like face toward +X.
      const toward = Math.max(0, n.x);
      v.x -= toward * radius * 0.22;
    }
    pos.setXYZ(i, v.x, v.y, v.z);
  }
  geo.computeVertexNormals();
  return geo;
}

function makeMoonMesh(name, { radius, color, seed, crater }) {
  const mat = new THREE.MeshStandardMaterial({
    map: makeRockTexture(seed),
    color,
    roughness: 0.92,
    metalness: 0.02,
    flatShading: true,
  });
  const mesh = new THREE.Mesh(
    makePotatoGeometry(radius, {
      stretch: name === "Phobos" ? 1.25 : 1.15,
      squash: name === "Phobos" ? 0.88 : 0.92,
      crater,
    }),
    mat
  );
  mesh.name = name;
  mesh.castShadow = false;
  mesh.receiveShadow = false;
  return mesh;
}

/**
 * @param {THREE.Scene} scene
 * @returns {{ update: (dt: number) => void, phobos: THREE.Object3D, deimos: THREE.Object3D }}
 */
export function addMarsMoons(scene) {
  const SIZE_EXAGGERATION = 15;

  // Semi-major axes (km) → scene units from Mars center.
  const phobosOrbitR = kmToScene(9376);
  const deimosOrbitR = kmToScene(23460);
  const phobosRadius = kmToScene(11.08) * SIZE_EXAGGERATION;
  const deimosRadius = kmToScene(6.2) * SIZE_EXAGGERATION;

  const phobosPivot = new THREE.Object3D();
  phobosPivot.name = "phobosOrbit";
  phobosPivot.rotation.x = THREE.MathUtils.degToRad(1.1);
  scene.add(phobosPivot);

  const phobos = makeMoonMesh("Phobos", {
    radius: phobosRadius,
    color: 0xb8a090,
    seed: 1,
    crater: true,
  });
  phobos.position.set(phobosOrbitR, 0, 0);
  phobosPivot.add(phobos);

  const deimosPivot = new THREE.Object3D();
  deimosPivot.name = "deimosOrbit";
  deimosPivot.rotation.x = THREE.MathUtils.degToRad(1.8);
  deimosPivot.rotation.z = THREE.MathUtils.degToRad(8);
  scene.add(deimosPivot);

  const deimos = makeMoonMesh("Deimos", {
    radius: deimosRadius,
    color: 0xc4b4a4,
    seed: 2,
    crater: false,
  });
  deimos.position.set(deimosOrbitR, 0, 0);
  deimosPivot.add(deimos);

  // Periods: Phobos 0.3189 d, Deimos 1.2625 d → ω_P / ω_D ≈ 3.96
  const deimosOmega = 0.08; // rad / second (demo-paced, not real-time)
  const phobosOmega = deimosOmega * (1.2625 / 0.3189);

  const marsCenter = new THREE.Vector3(0, 0, 0);

  return {
    phobos,
    deimos,
    update(dt) {
      phobosPivot.rotation.y += phobosOmega * dt;
      deimosPivot.rotation.y += deimosOmega * dt;
      // Tidally locked: same face toward Mars.
      phobos.lookAt(marsCenter);
      deimos.lookAt(marsCenter);
    },
  };
}

/**
 * Occasional faint meteors across the celestial sphere.
 * @returns {{ update: (dt: number) => void }}
 */
export function addMeteorShowers(scene) {
  const group = new THREE.Group();
  group.name = "meteors";
  scene.add(group);

  const pool = [];
  const POOL = 5;
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
    });
  }

  let cooldown = 1.5 + Math.random() * 3;
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

  function spawn(m) {
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
    m.length = 1.4 + Math.random() * 2.8;
    m.life = 0;
    m.maxLife = 0.5 + Math.random() * 0.6;
    m.speed = 20 + Math.random() * 24;
    m.line.visible = true;
    m.line.material.color.setHSL(0.08 + Math.random() * 0.08, 0.4, 0.88);
    m.line.material.opacity = 0;
  }

  return {
    update(dt) {
      cooldown -= dt;
      if (cooldown <= 0) {
        const idle = pool.find((m) => !m.line.visible);
        if (idle) spawn(idle);
        cooldown = 2.2 + Math.random() * 5;
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
        const fade = t < 0.2 ? t / 0.2 : t > 0.65 ? (1 - t) / 0.35 : 1;
        m.line.material.opacity = 0.65 * fade;
        tmp.copy(m.origin).addScaledVector(m.dir, m.speed * m.life);
        m.line.position.copy(tmp);
        m.line.quaternion.setFromUnitVectors(xAxis, m.dir);
        m.line.scale.set(m.length, 1, 1);
      }
    },
  };
}
